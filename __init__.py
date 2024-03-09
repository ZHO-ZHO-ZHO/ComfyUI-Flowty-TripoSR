import sys
from os import path
sys.path.insert(0, path.dirname(__file__))
from folder_paths import get_filename_list, get_full_path, get_save_image_path, get_output_directory
from comfy.model_management import get_torch_device
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
from PIL import Image
import numpy as np
import torch
import rembg
import os
from datetime import datetime


rembg_session = rembg.new_session()

def fill_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


class TripoSRModelLoader_Zho:
    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list("checkpoints"),),
                "chunk_size": ("INT", {"default": 8192, "min": 1, "max": 10000})
            }
        }

    RETURN_TYPES = ("TRIPOSR_MODEL",)
    FUNCTION = "load"
    CATEGORY = "TripoSR ZHO"

    def load(self, model, chunk_size):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_model:
            print("Loading TripoSR model")
            self.initialized_model = TSR.from_pretrained_custom(
                weight_path=get_full_path("checkpoints", model),
                config_path=path.join(path.dirname(__file__), "config.yaml")
            )
            self.initialized_model.renderer.set_chunk_size(chunk_size)
            self.initialized_model.to(device)

        return (self.initialized_model,)


class TripoSRSampler_Zho:

    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tpsr_model": ("TRIPOSR_MODEL",),
                "reference_image": ("IMAGE",),
                "do_remove_background": ("BOOLEAN", {"default": True}),
                "foreground_ratio": ("FLOAT", {"default": 0.85, "min": 0, "max": 1.0, "step": 0.01}),
                "geometry_extract_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 25.0, "min": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "sample"
    CATEGORY = "TripoSR ZHO"

    def sample(self, tpsr_model, reference_image, do_remove_background, foreground_ratio, geometry_extract_resolution, marching_cude_threshold):
        outputs = []

        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        with torch.no_grad():
            for image in reference_image:
                i = 255. * image.cpu().numpy()
                i = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                if do_remove_background:
                    i = i.convert("RGB")
                    i = remove_background(i, rembg_session)
                    i = resize_foreground(i, foreground_ratio)
                    i = fill_background(i)
                else:
                    i = i
                    if i.mode == "RGBA":
                        i = fill_background(i)
                scene_codes = tpsr_model([i], device)
                meshes = tpsr_model.extract_mesh(scene_codes, resolution=geometry_extract_resolution, threshold=marching_cude_threshold)
                outputs.append(meshes[0])

        return (outputs,)


class SaveTripoSR_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "filename_prefix": ("STRING", {"default": "TripoSR"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "TripoSR ZHO"

    def display(self, mesh, filename_prefix):
        saved = list()
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix,
                                                                                                get_output_directory())

        for (batch_number, single_mesh) in enumerate(mesh):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.obj"
            single_mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
            single_mesh.export(path.join(full_output_folder, file))
            saved.append({
                "filename": file,
                "type": "output",
                "subfolder": subfolder
            })

        return {"ui": {"mesh": saved}}

NODE_CLASS_MAPPINGS = {
    "TripoSRModelLoader_Zho": TripoSRModelLoader_Zho,
    "TripoSRSampler_Zho": TripoSRSampler_Zho,
    "SaveTripoSR_Zho": SaveTripoSR_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSRModelLoader_Zho": "TripoSR Model Loader",
    "TripoSRSampler_Zho": "TripoSR Sampler",
    "SaveTripoSR_Zho": "Save TripoSR"
}


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
