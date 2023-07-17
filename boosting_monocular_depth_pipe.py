from operator import getitem
from typing import Literal, Union
import torch
from torchvision.transforms import transforms
import numpy as np
import numpy.typing as npt
from PIL import Image
import cv2
from tqdm import tqdm

from BoostingMonocularDepth.lib.multi_depth_model_woauxi import RelDepthModel
from BoostingMonocularDepth.lib.net_tools import strip_prefix_if_present
from BoostingMonocularDepth.midas.models.midas_net import MidasNet
from BoostingMonocularDepth.midas.models.transforms import NormalizeImage, PrepareForNet, Resize
from BoostingMonocularDepth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from BoostingMonocularDepth.utils import (
    ImageAndPatches,
    apply_grid_patch,
    calculate_processing_resolution,
    generate_mask,
    get_GF_from_integral,
    rgb2gray,
)


class BoostingMonocularDepthPipeline(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        pix2pix_model_path: str,
        depth_estimator_model_path: str,
        depth_estimator: Literal['leres', 'midas'] = 'leres',
        r_threshold_value=0.2,
        scale_threshold_value=3,
        whole_size_threshold=3000,
        gpu_threshold=1600 - 32,
        output_depth_in_original_resolution=True,
    ):
        super().__init__()
        assert depth_estimator in ['leres', 'midas'], "Only the depth estimators 'leres' and 'midas' are supported"
        self.depth_estimator = depth_estimator
        self.device = device
        self.r_threshold_value = r_threshold_value
        self.scale_threshold = scale_threshold_value
        self.whole_size_threshold = whole_size_threshold  # R_max from the paper
        self.gpu_threshold = gpu_threshold  # Limit for the GPU (NVIDIA RTX 2080), can be adjusted
        self.max_res = np.inf
        self.output_depth_in_original_resolution = output_depth_in_original_resolution
        self.depth_estimator_model_path = depth_estimator_model_path

        # prepare pix2pix
        self.pix2pix_inference_dimensions = (1024, 1024)
        self.pix2pix_model = Pix2Pix4DepthModel(save_dir=pix2pix_model_path)
        self.pix2pix_model.load_networks("latest")
        self.pix2pix_model.eval()

        # prepare mask
        self.mask_org = generate_mask((3000, 3000))
        self.mask = self.mask_org.copy()

        self.factor = None  # see section 6 of main paper

        self.leres_receptive_field_size: int = 448
        self.leres_patch_size = 2 * self.leres_receptive_field_size
        self.midas_receptive_field_size: int = 384
        self.midas_patch_size = 2 * self.midas_receptive_field_size

        if self.depth_estimator == 'leres':
            # prepare depth estimation model LeRes
            leres_checkpoint = torch.load(self.depth_estimator_model_path)
            self.leres_model = RelDepthModel(backbone="resnext101")
            self.leres_model.load_state_dict(
                strip_prefix_if_present(leres_checkpoint["depth_model"], "module."),
                strict=True,
            )
            torch.cuda.empty_cache()
            self.leres_model.to(self.device)
            self.leres_model.eval()
            self.depth_estimator_receptive_field_size = self.leres_receptive_field_size
            self.depth_estimator_patch_size = self.leres_patch_size

        elif self.depth_estimator == 'midas':
            self.midas_model = MidasNet(self.depth_estimator_model_path, non_negative=True)
            self.midas_model.to(device)
            self.midas_model.eval()
            self.depth_estimator_receptive_field_size = self.midas_receptive_field_size
            self.depth_estimator_patch_size = self.midas_patch_size
        
        self.midas_scale: Union[float, None] = None
        self.midas_shift: Union[float, None] = None


    def forward(self, image: Image.Image, perform_boosting=True) -> npt.NDArray:
        image_array = np.asarray(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) / 255.0
        input_resolution = image_array.shape

        if not perform_boosting:
            prediction = self.single_estimate(image_array, inference_image_size=input_resolution[0])
            # if self.depth_estimator == 'midas':
            #     prediction = 1 - prediction
            
            return prediction

        # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
        # supplementary material.
        whole_image_optimal_size, patch_scale = calculate_processing_resolution(
            image_array,
            self.depth_estimator_receptive_field_size,
            self.r_threshold_value,
            self.scale_threshold,
            self.whole_size_threshold,
        )

        whole_estimate = self.double_estimate(
            image_array,
            depth_predictor_receptive_field_size=self.depth_estimator_receptive_field_size,
            optimal_inference_size=whole_image_optimal_size,
        )

        # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
        # small high-density regions of the image.
        self.factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / self.whole_size_threshold), 0.2)
        print("Adjust factor is:", 1 / self.factor)

        # Compute the default target resolution.
        if image_array.shape[0] > image_array.shape[1]:
            a = 2 * whole_image_optimal_size
            b = round(2 * whole_image_optimal_size * image_array.shape[1] / image_array.shape[0])
        else:
            a = round(2 * whole_image_optimal_size * image_array.shape[0] / image_array.shape[1])
            b = 2 * whole_image_optimal_size
        b = int(round(b / self.factor))
        a = int(round(a / self.factor))

        # recompute a, b and saturate to max res.
        if max(a, b) > self.max_res:
            print("Default Res is higher than max-res: Reducing final resolution")
            if image_array.shape[0] > image_array.shape[1]:
                a = self.max_res
                b = round(self.max_res * image_array.shape[1] / image_array.shape[0])
            else:
                a = round(self.max_res * image_array.shape[0] / image_array.shape[1])
                b = self.max_res
            b = int(b)
            a = int(a)

        image_array = cv2.resize(image_array, (b, a), interpolation=cv2.INTER_CUBIC)

        # Extract selected patches for local refinement
        base_size = self.depth_estimator_receptive_field_size * 2
        patch_set = self.generate_patches(image_array, base_size)

        print("Target resolution: ", image_array.shape)

        # Computing a scale in case user prompted to generate the results as the same resolution of the input.
        # Notice that our method output resolution is independent of the input resolution and this parameter will only
        # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
        # as the input.
        if self.output_depth_in_original_resolution == 1:
            mergein_scale = input_resolution[0] / image_array.shape[0]
            print("Dynamicly change merged-in resolution; scale:", mergein_scale)
        else:
            mergein_scale = 1

        image_and_patches = ImageAndPatches(
            root_dir=None,
            name=None,
            patches=patch_set,
            rgb_image=image_array,
            scale=mergein_scale,
        )
        whole_estimate_resized = cv2.resize(
            whole_estimate,
            (round(image_array.shape[1] * mergein_scale), round(image_array.shape[0] * mergein_scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        image_and_patches.set_base_estimate(whole_estimate_resized.copy())
        image_and_patches.set_updated_estimate(whole_estimate_resized.copy())

        print("\t Resulted depthmap res will be :", whole_estimate_resized.shape[:2])
        print("patchs to process: " + str(len(image_and_patches)))

        # Enumerate through all patches, generate their estimations and refining the base estimate.
        for patch_ind in range(len(image_and_patches)):
            # Get patch information
            patch = image_and_patches[patch_ind]  # patch object
            patch_rgb = patch["patch_rgb"]  # rgb patch
            patch_whole_estimate_base = patch["patch_whole_estimate_base"]  # corresponding patch from base
            rect = patch["rect"]  # patch size and location
            patch_id = patch["id"]  # patch ID
            org_size = patch_whole_estimate_base.shape  # the original size from the unscaled input
            print("\t processing patch", patch_ind, "|", rect)

            # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
            # field size of the network for patches to accelerate the process.
            patch_estimation = self.double_estimate(
                patch_rgb,
                self.depth_estimator_receptive_field_size,
                self.depth_estimator_patch_size,
            )

            patch_estimation = cv2.resize(
                patch_estimation,
                self.pix2pix_inference_dimensions,
                interpolation=cv2.INTER_CUBIC,
            )

            patch_whole_estimate_base = cv2.resize(
                patch_whole_estimate_base,
                self.pix2pix_inference_dimensions,
                interpolation=cv2.INTER_CUBIC,
            )

            # Merging the patch estimation into the base estimate using our merge network:
            # We feed the patch estimation and the same region from the updated base estimate to the merge network
            # to generate the target estimate for the corresponding region.
            self.pix2pix_model.set_input(patch_whole_estimate_base, patch_estimation)

            # Run merging network
            self.pix2pix_model.test()
            visuals = self.pix2pix_model.get_current_visuals()

            prediction_mapped = visuals["fake_B"]
            prediction_mapped = (prediction_mapped + 1) / 2
            prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

            mapped = prediction_mapped

            # We use a simple linear polynomial to make sure the result of the merge network would match the values of
            # base estimate
            p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
            merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

            merged = cv2.resize(merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

            # Get patch size and location
            w1 = rect[0]
            h1 = rect[1]
            w2 = w1 + rect[2]
            h2 = h1 + rect[3]

            # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
            # and resize it to our needed size while merging the patches.
            if self.mask.shape != org_size:
                self.mask = cv2.resize(self.mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

            to_be_merged_to = image_and_patches.estimation_updated_image

            # Update the whole estimation:
            # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
            # blending at the boundaries of the patch region.
            to_be_merged_to[h1:h2, w1:w2] = np.multiply(to_be_merged_to[h1:h2, w1:w2], 1 - self.mask) + np.multiply(
                merged, self.mask
            )
            image_and_patches.set_updated_estimate(to_be_merged_to)

        if self.output_depth_in_original_resolution == 1:
            final_estimation = cv2.resize(
                image_and_patches.estimation_updated_image,
                (input_resolution[1], input_resolution[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            final_estimation = image_and_patches.estimation_updated_image
        
        if self.depth_estimator == 'midas':
            final_estimation = 1 - final_estimation

        return final_estimation

    # Generate a double-input depth estimation
    def double_estimate(
        self,
        image_array: npt.NDArray,
        depth_predictor_receptive_field_size: int,
        optimal_inference_size: int,
    ) -> npt.NDArray:
        # Generate the low resolution estimation
        low_resolution_estimate = self.single_estimate(image_array, depth_predictor_receptive_field_size)
        # Resize to the inference size of merge network.
        low_resolution_estimate = cv2.resize(
            low_resolution_estimate,
            self.pix2pix_inference_dimensions,
            interpolation=cv2.INTER_CUBIC,
        )

        # Generate the high resolution estimation
        high_resolution_estimate = self.single_estimate(image_array, optimal_inference_size)
        # Resize to the inference size of merge network.
        high_resolution_estimate = cv2.resize(
            high_resolution_estimate,
            self.pix2pix_inference_dimensions,
            interpolation=cv2.INTER_CUBIC,
        )

        # Inference on the merge model
        self.pix2pix_model.set_input(low_resolution_estimate, high_resolution_estimate)
        self.pix2pix_model.test()
        visuals = self.pix2pix_model.get_current_visuals()
        prediction_mapped = visuals["fake_B"]
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
            torch.max(prediction_mapped) - torch.min(prediction_mapped)
        )
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        return prediction_mapped

    def single_estimate(self, image_array: npt.NDArray, inference_image_size: int) -> npt.NDArray:
        if inference_image_size > self.gpu_threshold:
            print(" \t \t DEBUG| GPU THRESHOLD REACHED", inference_image_size, "--->", self.gpu_threshold)
            inference_image_size = self.gpu_threshold

        if self.depth_estimator == 'midas':
            return self.estimate_midas(image_array, midas_inference_size=inference_image_size)
        return self.estimate_leres(image_array, leres_inference_size=inference_image_size)

    def estimate_leres(self, image_array: npt.NDArray, leres_inference_size: int) -> npt.NDArray:
        # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
        rearranged_image_array_copy = image_array[:, :, ::-1].copy()  # TODO: why is this necessary?
        leres_inference_dimensions = (leres_inference_size, leres_inference_size)
        resized_rearranged_image_array = cv2.resize(rearranged_image_array_copy, leres_inference_dimensions)
        image_tensor = self.scale_torch(resized_rearranged_image_array)[None, :, :, :]

        # Forward pass
        with torch.no_grad():
            prediction = self.leres_model.inference(image_tensor)

        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_CUBIC)

        return prediction
    
    def estimate_midas(self, image_array: npt.NDArray, midas_inference_size: int) -> npt.NDArray:
       # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

        transform = transforms.Compose(
            [
                Resize(
                    midas_inference_size,
                    midas_inference_size,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        img_input = transform({"image": image_array})["image"]

        # Forward pass
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            disparity_prediction = self.midas_model.forward(sample)

        disparity_prediction = disparity_prediction.squeeze().cpu().numpy()
        disparity_prediction = cv2.resize(disparity_prediction, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_CUBIC)

        new_min = 0.5
        new_max = 4
        scaled_disparity = ((new_max - new_min) * (disparity_prediction - disparity_prediction.min())) / (disparity_prediction.max() - disparity_prediction.min()) + new_min

        depth = depth_prediction = 1 / scaled_disparity

        # depth = np.nan_to_num(depth)

        # # Inverted Normalization
        # midas_scale_and_shift_unknown = self.midas_scale == None and self.midas_shift == None
        # if midas_scale_and_shift_unknown:
        #     self.midas_scale = depth.max() - depth.min()
        #     self.midas_shift = depth.min()

        # if self.midas_scale and np.abs(self.midas_scale) > np.finfo("float").eps:
        #     normalized_depth = ((new_max - new_min) * (depth - self.midas_shift )) / self.midas_scale + new_min
        # else:
        #     normalized_depth = 0

        return depth

    def scale_torch(self, image_array: npt.NDArray) -> torch.Tensor:
        """
        Scale the image and output it in torch.tensor.
        :param image_array: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :return: img. [C, H, W]
        """
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, :, :]
        if image_array.shape[2] == 3:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
            image_array = transform(image_array.astype(np.float32))
        else:
            image_array = image_array.astype(np.float32)
            image_array = torch.from_numpy(image_array)

        return image_array

    def generate_patches(self, image_array: npt.NDArray, base_size: int):
        # Compute the gradients as a proxy of the contextual cues.
        img_gray = rgb2gray(image_array)
        whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(
            cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        )

        threshold = whole_grad[whole_grad > 0].mean()
        whole_grad[whole_grad < threshold] = 0

        # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
        gf = whole_grad.sum() / len(whole_grad.reshape(-1))
        grad_integral_image = cv2.integral(whole_grad)

        # Variables are selected such that the initial patch size would be the receptive field size
        # and the stride is set to 1/3 of the receptive field size.
        blsize = int(round(base_size / 2))
        stride = int(round(blsize * 0.75))

        # Get initial Grid
        patch_bound_list = apply_grid_patch(blsize, stride, image_array, [0, 0, 0, 0])

        # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
        # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
        print("Selecting patchs ...")
        patch_bound_list = self.adaptive_selection(grad_integral_image, patch_bound_list, gf)

        # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
        # patch
        patch_set = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], "size"), reverse=True)

        return patch_set

    # Adaptively select patches
    def adaptive_selection(self, integral_grad, patch_bound_list, gf):
        patchlist = {}
        count = 0
        height, width = integral_grad.shape

        search_step = int(32 / self.factor)

        # Go through all patches
        for c in range(len(patch_bound_list)):
            # Get patch
            bbox = patch_bound_list[str(c)]["rect"]

            # Compute the amount of gradients present in the patch from the integral image.
            cgf = get_GF_from_integral(integral_grad, bbox) / (bbox[2] * bbox[3])

            # Check if patching is beneficial by comparing the gradient density of the patch to
            # the gradient density of the whole image
            if cgf >= gf:
                bbox_test = bbox.copy()
                patchlist[str(count)] = {}

                # Enlarge each patch until the gradient density of the patch is equal
                # to the whole image gradient density
                while True:
                    bbox_test[0] = bbox_test[0] - int(search_step / 2)
                    bbox_test[1] = bbox_test[1] - int(search_step / 2)

                    bbox_test[2] = bbox_test[2] + search_step
                    bbox_test[3] = bbox_test[3] + search_step

                    # Check if we are still within the image
                    if (
                        bbox_test[0] < 0
                        or bbox_test[1] < 0
                        or bbox_test[1] + bbox_test[3] >= height
                        or bbox_test[0] + bbox_test[2] >= width
                    ):
                        break

                    # Compare gradient density
                    cgf = get_GF_from_integral(integral_grad, bbox_test) / (bbox_test[2] * bbox_test[3])
                    if cgf < gf:
                        break
                    bbox = bbox_test.copy()

                # Add patch to selected patches
                patchlist[str(count)]["rect"] = bbox
                patchlist[str(count)]["size"] = bbox[2]
                count = count + 1

        # Return selected patches
        return patchlist
    
    @torch.enable_grad()
    def estimate_fined_tuned_midas_depth(self,
        image: Image.Image,
        inpainting_mask: torch.Tensor,
        rendered_depth: torch.Tensor,
        number_training_epochs = 300,
        learning_rate = 1e-7,
    ) -> npt.NDArray:
        image_array = np.asarray(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) / 255.0
        transform = transforms.Compose(
            [
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        unnormalized_rendered_disparity = rendered_depth * self.midas_scale + self.midas_shift 
        input_image = transform({"image": image_array})["image"]

        self.midas_model = MidasNet(self.depth_estimator_model_path, non_negative=True)
        self.midas_model.train()
        optimizer = torch.optim.Adam(self.midas_model.parameters(), lr=learning_rate)

        self.midas_model.to(self.device)

        def loss_function(
            predicted_disparity: torch.Tensor,
            rendered_disparity: torch.Tensor,
        ) -> torch.Tensor:
            loss = torch.linalg.matrix_norm(rendered_disparity * ~inpainting_mask - predicted_disparity * ~inpainting_mask, ord=1)
            return loss

        # Fine tune
        progress_bar = tqdm(range(number_training_epochs))
        for epoch in progress_bar:
            input = torch.from_numpy(input_image).to(self.device).unsqueeze(0)
            optimizer.zero_grad()

            disparity_prediction = self.midas_model.forward(input)

            loss = loss_function(disparity_prediction, unnormalized_rendered_disparity)
            loss.backward()

            optimizer.step()
            progress_bar.set_description(f"fine-tuning epoch [{epoch + 1}/{number_training_epochs}]")
        
        # Predict
        self.midas_model.eval()
        with torch.no_grad():
            input = torch.from_numpy(input_image).to(self.device).unsqueeze(0)
            disparity_prediction = self.midas_model.forward(input) 

        disparity_prediction = disparity_prediction.squeeze().cpu().numpy()
        disparity_prediction = cv2.resize(disparity_prediction, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_CUBIC)

        depth = depth_prediction = disparity_prediction

        # Inverted Normalization
        if self.midas_scale and np.abs(self.midas_scale) > np.finfo("float").eps:
            normalized_depth = (depth - self.midas_shift ) / self.midas_scale
        else:
            normalized_depth = 0

        return normalized_depth
