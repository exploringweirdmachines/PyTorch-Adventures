import os
import torch
import numpy as np
from PIL import Image
from dataset import image_transforms

def count_num_params(model):

    total = 0
    for param in model.parameters():
        total += param.numel()
    
    suffixes = ['', 'K', 'M', 'B', 'T']
    
    # Find the magnitude of the number (i.e., how many powers of 1000 it has)
    magnitude = 0
    while abs(total) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        total /= 1000.0
    
    # Format the number with 1 decimal place
    return f"{total:.1f}{suffixes[magnitude]}"


def load_val_images(path_to_image_folder, 
                    img_size,
                    device, 
                    dtype):

    image_files = os.listdir(path_to_image_folder)
    path_to_imgs = [os.path.join(path_to_image_folder, file) for file in image_files]
    
    val_img_transforms = image_transforms(img_size=img_size, 
                                          train=False)
    
    val_images = [Image.open(path).convert("RGB") for path in path_to_imgs]
    val_images = torch.stack([val_img_transforms(img) for img in val_images])

    val_images = val_images.to(device)

    # Cast images to the correct precision type
    weight_dtype = torch.float32 
    if dtype == "fp16":
        weight_dtype = torch.float16
    elif dtype == "bf16":
        weight_dtype = torch.bfloat16
    
    val_images = val_images.to(weight_dtype)

    return val_images

def save_generated_images(original_images, 
                          generated_image_tensors, 
                          path_to_save_folder, 
                          step,
                          accelerator):
    
    ### Create Folder if it doesnt Exist ###
    if not os.path.isdir(path_to_save_folder):
        accelerator.print(f"Creating Folder {path_to_save_folder} to save Reconstructions")
        os.makedirs(path_to_save_folder)

    ### Clamp Output Between [-1 to 1] and rescale back to [0 to 255] ###
    generated_image_tensors = torch.clamp(generated_image_tensors, -1., 1.)
    generated_image_tensors = (generated_image_tensors + 1) / 2
    generated_image_tensors = generated_image_tensors.cpu().permute(0,2,3,1).numpy()
    generated_image_tensors = (255 * generated_image_tensors).astype(np.uint8)
    gen_imgs = [Image.fromarray(img).convert("RGB") for img in generated_image_tensors]

    ### Original Images have been scaled to [-1 to 1], rescale back to [0 to 255] ###
    original_images = (original_images + 1) / 2
    original_images = original_images.cpu().permute(0,2,3,1).numpy()
    original_images = (255 * original_images).astype(np.uint8)
    orig_imgs = [Image.fromarray(img).convert("RGB") for img in original_images]

    ### Concat Images (so we can compare real vs reconstruction) ###
    img_width = orig_imgs[0].width
    img_height = orig_imgs[0].height
    combined_images = []
    for orig_img, gen_img in zip(orig_imgs, gen_imgs):
        combined_img = Image.new(mode="RGB", size=(img_width, 2*img_height))
        combined_img.paste(orig_img, (0,0))
        combined_img.paste(gen_img, (0,img_height))
        combined_images.append(combined_img)

    ### Concatenate All Samples Together ###
    final_image = Image.new(mode="RGB", size=(img_width*len(combined_images), 2*img_height))
    x_offset = 0
    for img in combined_images:
        final_image.paste(img, (x_offset,0))
        x_offset += img_width
    
    ### Save Output ###
    path_to_save = os.path.join(path_to_save_folder, f"iteration_{step}.png")
    final_image.save(path_to_save)

        
