from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
import cv2
import glob
import numpy as np
import os
import torch
import warnings

from pydantic import BaseModel

from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

app = FastAPI()


FOLDER_PATH = os.getenv("FOLDER_PATH")
MOUNT_PATH = os.getenv("MOUNT_PATH")
# Global variables
restorer = None

# Function to check and initialize the restorer
def initialize_restorer():
    global restorer

    # Default parameters (can be adjusted as needed)
    version = '1.3'  # GFPGAN model version
    upscale = 2      # Upscaling factor
    bg_upsampler_name = 'realesrgan'  # Background upsampler
    bg_tile = 400    # Tile size for background upsampler
    weight = 0.5     # Adjustable weight

    # ------------------------ Set up background upsampler ------------------------
    if bg_upsampler_name == 'realesrgan':
        if not torch.cuda.is_available():  # CPU mode
            warnings.warn('RealESRGAN is slow on CPU. The background will not be upsampled.')
            bg_upsampler = None
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # Set False for CPU mode
    else:
        bg_upsampler = None

    # ------------------------ Set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # Determine model path
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # Use URL if model is not found locally
        model_path = url

    # Initialize GFPGANer
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

# Function to restore images in a given folder and its subfolders
def restore_images(folder_path):
    # Find all image files in the folder and subfolders
    img_list = []
    for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp'):
        img_list.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))

    if not img_list:
        print(f'No images found in {folder_path}')
        return

    # Process each image
    for img_path in img_list:
        print(f'Processing {img_path} ...')

        # Read the image
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if input_img is None:
            print(f'Failed to read image {img_path}')
            continue

        # Restore the image
        try:
            _, _, restored_img = restorer.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5)
        except Exception as e:
            print(f'Failed to restore image {img_path}. Error: {e}')
            continue

        # Overwrite the original image with the restored image
        if restored_img is not None:
            imwrite(restored_img, img_path)
        else:
            print(f'Failed to restore image {img_path}')

# Pydantic model for request body
class FilePath(BaseModel):
    file_path: str

# Function to restore a single image
def restore_single_image(path):
    # Read the image
    input_img = cv2.imread(path, cv2.IMREAD_COLOR)
    if input_img is None:
        return {"error": f"Failed to read image {path}"}

    # Restore the image
    try:
        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)
    except Exception as e:
        return {"error": f"Failed to restore image {path}. Error: {str(e)}"}

    # Overwrite the original image with the restored image
    if restored_img is not None:
        imwrite(restored_img, path)
        return {"message": f"Image {path} has been restored"}
    else:
        return {"error": f"Failed to restore image {path}"}

# API endpoint that accepts a case_id and processes the corresponding images
@app.post("/restore/{case_id}")
def restore(case_id: str):
    prefix_path = FOLDER_PATH
    folder_path = os.path.join(prefix_path, case_id)

    if not os.path.exists(folder_path):
        return {"error": f"Folder {folder_path} does not exist"}

    # Call the restore_images function
    restore_images(folder_path)

    return {"message": f"Images in folder {folder_path} have been restored"}

# New API endpoint that accepts a file_path and processes the image
@app.post("/restore-file")
def restore_file(file_path: FilePath):
    path = file_path.file_path
    if MOUNT_PATH !="" and MOUNT_PATH:
      local_path = path.replace(MOUNT_PATH, FOLDER_PATH)
    else:
      local_path = path

    if not os.path.exists(local_path):
        return {"error": f"File {local_path} does not exist"}

    # Call the restore_single_image function
    result = restore_single_image(local_path)
    return result

# Initialize the restorer when the application starts
@app.on_event("startup")
def startup_event():
    initialize_restorer()
