import torch
from torchvision.transforms import Compose
from torchvision.transforms import transforms
import numpy as np
import numpy.typing as npt
from PIL import Image
import cv2

from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from pix2pix.options.test_options import TestOptions
from run import double_estimate, generate_patches
from utils import ImageAndPatches, calculate_processing_resolution, generate_mask


class BoostingMonocularDepthPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # prepare pix2pix
        pix2pix_inference_settings = TestOptions().parse()
        self.pix2pix_inference_dimensions = (1024, 1024)
        self.pix2pix_model = Pix2Pix4DepthModel(pix2pix_inference_settings)
        self.pix2pix_model.save_dir = './pix2pix/checkpoints/mergemodel'  # TODO: add parameter for path
        self.pix2pix_model.load_networks("latest")
        self.pix2pix_model.eval()

        # prepare depth estimation model LeRes
        self.leres_receptive_field_size: int = 448
        self.leres_patch_size = 2 * self.leres_receptive_field_size
        leres_model_path = "res101.pth"  # TODO: add parameter for path
        leres_checkpoint = torch.load(leres_model_path)
        self.leres_model = RelDepthModel(backbone="resnext101")
        self.leres_model.load_state_dict(
            strip_prefix_if_present(leres_checkpoint["depth_model"], "module."),
            strict=True,
        )
        torch.cuda.empty_cache()
        self.leres_model.to(self.device)  # TODO: add parameter for device
        self.leres_model.eval()

        # prepare mask
        self.mask_org = generate_mask(
            (3000, 3000)
        )  # TODO: rename function and make method of class
        self.mask = self.mask_org.copy()

        self.r_threshold_value = 0.2  # TODO: add parameter for r
        self.scale_threshold = 3
        self.factor = None  # TODO: what is this?
        self.whole_size_threshold = 3000  # R_max from the paper
        self.gpu_threshold = 1600 - 32  # Limit for the GPU (NVIDIA RTX 2080), can be adjusted
        self.max_res = np.inf
        self.output_resolution = 1 # TODO: make this a boolean
        # TODO: support other models as well
        self.depth_estimator_receptive_field_size = self.leres_receptive_field_size
        self.depth_estimator_patch_size = self.leres_patch_size

    def forward(self, image: Image):
        image_array = np.asarray(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) / 255.0
        input_resolution = image_array.shape

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
        print('Adjust factor is:', 1 / self.factor)

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
        if max(a,b) > self.max_res:
            print('Default Res is higher than max-res: Reducing final resolution')
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
        patch_set = generate_patches(image_array, base_size)

        print('Target resolution: ', image_array.shape)

        # Computing a scale in case user prompted to generate the results as the same resolution of the input.
        # Notice that our method output resolution is independent of the input resolution and this parameter will only
        # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
        # as the input.
        if self.output_resolution == 1:
            mergein_scale = input_resolution[0] / image_array.shape[0]
            print('Dynamicly change merged-in resolution; scale:', mergein_scale)
        else:
            mergein_scale = 1

        image_and_patches = ImageAndPatches(
            root_dir= None,
            name=None,
            patches=patch_set,
            image=image_array,
            scale=mergein_scale,
        )
        whole_estimate_resized = cv2.resize(
            whole_estimate,
            (round(image_array.shape[1] * mergein_scale), round(image_array.shape[0] * mergein_scale)),
            interpolation=cv2.INTER_CUBIC
        )
        image_and_patches.set_base_estimate(whole_estimate_resized.copy())
        image_and_patches.set_updated_estimate(whole_estimate_resized.copy())

        print('\t Resulted depthmap res will be :', whole_estimate_resized.shape[:2])
        print('patchs to process: ' + str(len(image_and_patches)))

        # Enumerate through all patches, generate their estimations and refining the base estimate.
        for patch_ind in range(len(image_and_patches)):
            
            # Get patch information
            patch = image_and_patches[patch_ind] # patch object
            patch_rgb = patch['patch_rgb'] # rgb patch
            patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
            rect = patch['rect'] # patch size and location
            patch_id = patch['id'] # patch ID
            org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
            print('\t processing patch', patch_ind, '|', rect)

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

            prediction_mapped = visuals['fake_B']
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
            if mask.shape != org_size:
                mask = cv2.resize(self.mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)

            to_be_merged_to = image_and_patches.estimation_updated_image

            # Update the whole estimation:
            # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
            # blending at the boundaries of the patch region.
            to_be_merged_to[h1:h2, w1:w2] = np.multiply(to_be_merged_to[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
            image_and_patches.set_updated_estimate(to_be_merged_to)

        if self.output_resolution == 1:
            final_estimation = cv2.resize(
                image_and_patches.estimation_updated_image,
                (input_resolution[1], input_resolution[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            final_estimation = image_and_patches.estimation_updated_image

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
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = (
            (prediction_mapped - torch.min(prediction_mapped))
            / (torch.max(prediction_mapped) - torch.min(prediction_mapped))
        )
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        return prediction_mapped

    
    def single_estimate(self, image_array: npt.NDArray, inference_image_size: int) -> npt.NDArray:
        if inference_image_size > self.gpu_threshold:
            print(" \t \t DEBUG| GPU THRESHOLD REACHED", inference_image_size, '--->', self.gpu_threshold)
            inference_image_size = self.gpu_threshold

        # TODO: support other network types as well
        return self.estimate_leres(image_array, leres_inference_size=inference_image_size)


    def estimate_leres(self, image_array: npt.NDArray, leres_inference_size: int) -> npt.NDArray:
        # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
        rearranged_image_array_copy = image_array[:, :, ::-1].copy() # TODO: why is this necessary?
        leres_inference_dimensions = (leres_inference_size, leres_inference_size)
        resized_rearranged_image_array = cv2.resize(rearranged_image_array_copy, leres_inference_dimensions)
        image_tensor = self.scale_torch(resized_rearranged_image_array)[None, :, :, :]

        # Forward pass
        with torch.no_grad():
            prediction = self.leres_model.inference(image_tensor)

        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_CUBIC)

        return prediction
    

    def scale_torch(self, image_array: npt.NDArray) -> torch.Tensor:
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, :, :]
        if image_array.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
            image_array = transform(image_array.astype(np.float32))
        else:
            image_array = image_array.astype(np.float32)
            image_array = torch.from_numpy(image_array)

        return image_array
