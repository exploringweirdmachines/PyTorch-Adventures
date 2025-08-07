import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm import tqdm 

from model import SRNet, Discriminator
from loss import VGGPerceptualLoss
from dataset import SRImageDataset

accelerator = Accelerator()

### TRAINING CONFIG ###
num_iterations = 100000
save_gen_iters = 2500
batch_size = 32
image_size = 96
num_workers = 32
upsampling_factor = 4
generator_lr = 1e-4
discriminator_lr = 1e-4
path_to_train_data = "/mnt/datadrive/data/ImageNet/train"
path_to_test_data = "/mnt/datadrive/data/ImageNet/validation"
path_to_gens = "work_dir/"
device = accelerator.device

### Data Prep ###
dataset = SRImageDataset(root_dir=path_to_train_data, 
                         hr_size=image_size, 
                         scale_factor=upsampling_factor)

dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=num_workers,
                        pin_memory=True)

val_dataset = SRImageDataset(root_dir=path_to_test_data, 
                             hr_size=image_size, 
                             scale_factor=upsampling_factor, 
                             train_transform=False)

val_loader = iter(DataLoader(val_dataset, batch_size=1, shuffle=True))

### Load Models ###
generator = SRNet().to(device)
discriminator = Discriminator().to(device)

### Convert Batchnorm to SyncBatchNorm (For MultiGPU Training) ###
if accelerator.num_processes > 1:
    discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

### Load Perceptual Loss ###
vgg_loss = VGGPerceptualLoss().to(device)

### Load Optimizers ###
optimizer_g = optim.Adam(generator.parameters(), lr=generator_lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=discriminator_lr)

generator, discriminator, vgg_loss, optimizer_g, optimizer_g, dataloader = accelerator.prepare(
    generator, discriminator, vgg_loss, optimizer_g, optimizer_g, dataloader
)

train = True
completed_steps = 0
pbar = tqdm(range(num_iterations))

while train:

    for hr_images, lr_images in dataloader:
  
        hr_images, lr_images = hr_images.to(device), lr_images.to(device)

        batch_size = hr_images.shape[0]

        ### Create Real and Fake Labels ###
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        ### Generate Fake Upsampled Images ###
        upsampled_images = generator(lr_images)
        
        ### Pass Real and Fake Images to Discriminator ###
        real_preds = discriminator(hr_images)
        fake_preds = discriminator(upsampled_images.detach())

        ### Compute Discriminator Loss ###
        disc_loss_real = F.binary_cross_entropy_with_logits(real_preds.squeeze(-1), real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(fake_preds.squeeze(-1), fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        ### Update Discriminator ###
        optimizer_d.zero_grad()
        disc_loss.backward()
        optimizer_d.step()

        ### Train Generator with GAN + Perceptual + Pixel Loss ###
        fake_preds = discriminator(upsampled_images)
        adv_loss = F.binary_cross_entropy_with_logits(fake_preds.squeeze(-1), real_labels)
        perceptual_loss = vgg_loss(upsampled_images, hr_images)
        pixel_loss = torch.mean((upsampled_images - hr_images)**2)

        ### Total Generator Loss ###
        gen_loss = pixel_loss + 0.001 * adv_loss + 0.006 * perceptual_loss

        ### Update Generator ###
        optimizer_g.zero_grad()
        gen_loss.backward()
        optimizer_g.step()

        ### Generate Images Every Save Steps ###
        if completed_steps % save_gen_iters == 0:
            accelerator.print("Saving Generation Sample")

            generator.eval()
            discriminator.eval()

            sample_hr_image, sample_lr_image = next(val_loader)

            with torch.no_grad():
                gen_image = generator(sample_lr_image)

            sample_lr_image = ((sample_lr_image * 0.5) + 0.5).squeeze(0).permute(1,2,0).cpu().numpy()
            sample_hr_image = ((sample_hr_image * 0.5) + 0.5).squeeze(0).permute(1,2,0).cpu().numpy()
            gen_image = ((gen_image * 0.5) + 0.5).squeeze(0).permute(1,2,0).cpu().numpy()

            if accelerator.is_main_process:
                # Create subplots with 1 row and 3 columns
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))

                # Display each image
                axes[0].imshow(sample_lr_image, cmap='gray')
                axes[0].set_title("LR")
                axes[0].axis('off')

                axes[1].imshow(sample_hr_image, cmap='gray')
                axes[1].set_title("HR")
                axes[1].axis('off')

                axes[2].imshow(gen_image, cmap='gray')
                axes[2].set_title("GEN")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(path_to_gens, f"iter_{completed_steps}.png"))

            generator.train()
            discriminator.train()

        accelerator.wait_for_everyone()

        ### Iter Steps ###
        completed_steps += 1
        pbar.update(1)

        if completed_steps >= num_iterations:
            train = False
            print("Completed Training")
            break




                



