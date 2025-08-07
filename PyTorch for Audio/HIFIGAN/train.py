# import os
# import argparse
# import torch
# from transformers import set_seed
# from accelerate import Accelerator

# from dataset import MelDataset
# from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, \
#     feature_loss, generator_loss, discriminator_loss

# def parse_args():

#     parser = argparse.ArgumentParser()

#     ### SETUP CONFIG ###
#     parser.add_argument("--experiment_name", type=str, required=True)
#     parser.add_argument("--working_directory", type=str, required=True)
#     parser.add_argument("--path_to_train_manifest", type=str, required=True)
#     parser.add_argument("--path_to_val_manifest", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)

#     ### TRAINING CONFIG ###
#     parser.add_argument("--training_epochs", type=int, default=3100)
#     parser.add_argument("--evaluation_iters", type=int, default=500)
#     parser.add_argument("--checkpoint_iters", type=int, default=5000)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--learning_rate", type=float, default=0.0002)
#     parser.add_argument("--beta1", type=float, default=0.8)
#     parser.add_argument("--beta2", type=float, default=0.99)
#     parser.add_argument("--lr_decay", type=float, default=0.999)
#     parser.add_argument("--lambda_mel", type=float, default=45.)
#     parser.add_argument("--lambda_feature_mapping", type=float, default=2.)


#     ### MODEL CONFIG ###
#     parser.add_argument("--upsample_rates", type=int, nargs='+', default=(8, 8, 2, 2))
#     parser.add_argument("--upsample_kernel_sizes", type=int, nargs='+', default=(16, 16, 4, 4))
#     parser.add_argument("--upsample_initial_channel", type=int, default=512)
#     parser.add_argument("--resblock_kernel_sizes", type=int, nargs='+', default=(3, 7, 11))
#     parser.add_argument(
#         "--resblock_dilation_sizes",
#         type=eval,
#         default="((1, 3, 5), (1, 3, 5), (1, 3, 5))",
#         help="Nested tuple, e.g., '((1,3,5), (1,3,5), (1,3,5))'"
#     )
#     parser.add_argument("--mpd_periods", type=int, nargs='+', default=(2, 3, 5, 7, 11))
#     parser.add_argument("--msd_num_downsamples", type=int, default=2)

#     ### DATASET CONFIG ###
#     parser.add_argument("--sampling_rate", type=int, default=22050)
#     parser.add_argument("--segment_size", type=int, default=8192)
#     parser.add_argument("--num_mels", type=int, default=80)
#     parser.add_argument("--n_fft", type=int, default=1024)
#     parser.add_argument("--window_size", type=int, default=1024)
#     parser.add_argument("--hop_size", type=int, default=256)
#     parser.add_argument("--fmin", type=int, default=0)
#     parser.add_argument("--fmax", type=int, default=8000)
#     parser.add_argument("--fmax_loss", type=int, default=None)
#     parser.add_argument("--n_cache_reuse", type=int, default=1)
#     parser.add_argument("--num_workers", type=int, default=16)
#     parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

#     return parser.parse_args()

# ### Parser Arguments ###
# args = parse_args()

# ### Set Seed ###
# if args.seed is not None:
#     set_seed(args.seed)

# ### Init Accelerator ###
# path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
# accelerator = Accelerator(project_dir=path_to_experiment,
#                           log_with="wandb" if args.log_wandb else None)

# ### Load Model ###
# generator = Generator(args)
# mpd = MultiPeriodDiscriminator()
# msd = MultiScaleDiscriminator()

# ### Load Checkpoint ###

# ### Load Optimizer ###
# optim_g = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate, betas=[args.beta1, args.beta2])
# optim_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=args.learning_rate, betas=[args.beta1, args.beta2])

# ### Load Optimizer State ###

# ### Load Scheduler ###
# scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay, last_epoch=-1)
# scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.lr_decay, last_epoch=-1)

# ### Load Dataset ###
# # trainset = Me




# import os
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from accelerate import Accelerator
# from transformers import set_seed   
# from tqdm import tqdm

# from dataset import MelDataset, compute_mel_spectrogram
# from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
# from utils import print_log

# def parse_args():

#     parser = argparse.ArgumentParser()

#     ### SETUP CONFIG ###
#     parser.add_argument("--experiment_name", type=str, required=True)
#     parser.add_argument("--working_directory", type=str, required=True)
#     parser.add_argument("--path_to_train_manifest", type=str, required=True)
#     parser.add_argument("--path_to_val_manifest", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)

#     ### TRAINING CONFIG ###
#     parser.add_argument("--training_iterations", type=int, default=500000)
#     parser.add_argument("--evaluation_iters", type=int, default=500)
#     parser.add_argument("--checkpoint_iters", type=int, default=5000)
#     parser.add_argument("--batch_size_per_gpu", type=int, default=16)
#     parser.add_argument("--learning_rate", type=float, default=0.0002)
#     parser.add_argument("--beta1", type=float, default=0.8)
#     parser.add_argument("--beta2", type=float, default=0.99)
#     parser.add_argument("--lr_decay", type=float, default=0.999)
#     parser.add_argument("--lambda_mel", type=float, default=45.)
#     parser.add_argument("--lambda_feature_mapping", type=float, default=2.)


#     ### MODEL CONFIG ###
#     parser.add_argument("--upsample_rates", type=int, nargs='+', default=(8, 8, 2, 2))
#     parser.add_argument("--upsample_kernel_sizes", type=int, nargs='+', default=(16, 16, 4, 4))
#     parser.add_argument("--upsample_initial_channel", type=int, default=512)
#     parser.add_argument("--residual_block_kernel_sizes", type=int, nargs='+', default=(3, 7, 11))
#     parser.add_argument(
#         "--residual_block_dilation_sizes",
#         type=eval,
#         default="((1, 3, 5), (1, 3, 5), (1, 3, 5))",
#         help="Nested tuple, e.g., '((1,3,5), (1,3,5), (1,3,5))'"
#     )
#     parser.add_argument("--mpd_periods", type=int, nargs='+', default=(2, 3, 5, 7, 11))
#     parser.add_argument("--msd_num_downsamples", type=int, default=2)

#     ### DATASET CONFIG ###
#     parser.add_argument("--sampling_rate", type=int, default=22050)
#     parser.add_argument("--segment_size", type=int, default=8192)
#     parser.add_argument("--num_mels", type=int, default=80)
#     parser.add_argument("--n_fft", type=int, default=1024)
#     parser.add_argument("--window_size", type=int, default=1024)
#     parser.add_argument("--hop_size", type=int, default=256)
#     parser.add_argument("--fmin", type=int, default=0)
#     parser.add_argument("--fmax", type=int, default=8000)
#     parser.add_argument("--fmax_loss", type=int, default=None)
#     parser.add_argument("--n_cache_reuse", type=int, default=1)
#     parser.add_argument("--num_workers", type=int, default=16)
#     parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

#     return parser.parse_args()

# def parse_args():

#     parser = argparse.ArgumentParser()

#     ### SETUP CONFIG ###
#     parser.add_argument("--experiment_name", type=str, required=True)
#     parser.add_argument("--working_directory", type=str, required=True)
#     parser.add_argument("--path_to_train_manifest", type=str, required=True)
#     parser.add_argument("--path_to_val_manifest", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=None)

#     ### TRAINING CONFIG ###
#     parser.add_argument("--training_iterations", type=int, default=500000)
#     parser.add_argument("--evaluation_iters", type=int, default=500)
#     parser.add_argument("--checkpoint_iters", type=int, default=5000)
#     parser.add_argument("--batch_size_per_gpu", type=int, default=16)
#     parser.add_argument("--learning_rate", type=float, default=0.0002)
#     parser.add_argument("--beta1", type=float, default=0.8)
#     parser.add_argument("--beta2", type=float, default=0.99)
#     parser.add_argument("--lr_decay", type=float, default=0.999)
#     parser.add_argument("--lambda_mel", type=float, default=45.)
#     parser.add_argument("--lambda_feature_mapping", type=float, default=2.)


#     ### MODEL CONFIG ###
#     parser.add_argument("--upsample_rates", type=int, nargs='+', default=(8, 8, 2, 2))
#     parser.add_argument("--upsample_kernel_sizes", type=int, nargs='+', default=(16, 16, 4, 4))
#     parser.add_argument("--upsample_initial_channel", type=int, default=512)
#     parser.add_argument("--resblock_kernel_sizes", type=int, nargs='+', default=(3, 7, 11))
#     parser.add_argument(
#         "--resblock_dilation_sizes",
#         type=eval,
#         default="((1, 3, 5), (1, 3, 5), (1, 3, 5))",
#         help="Nested tuple, e.g., '((1,3,5), (1,3,5), (1,3,5))'"
#     )
#     parser.add_argument("--mpd_periods", type=int, nargs='+', default=(2, 3, 5, 7, 11))
#     parser.add_argument("--msd_num_downsamples", type=int, default=2)

#     ### DATASET CONFIG ###
#     parser.add_argument("--sampling_rate", type=int, default=22050)
#     parser.add_argument("--segment_size", type=int, default=8192)
#     parser.add_argument("--num_mels", type=int, default=80)
#     parser.add_argument("--n_fft", type=int, default=1024)
#     parser.add_argument("--window_size", type=int, default=1024)
#     parser.add_argument("--hop_size", type=int, default=256)
#     parser.add_argument("--fmin", type=int, default=0)
#     parser.add_argument("--fmax", type=int, default=8000)
#     parser.add_argument("--fmax_loss", type=int, default=None)
#     parser.add_argument("--n_cache_reuse", type=int, default=1)
#     parser.add_argument("--num_workers", type=int, default=16)
#     parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

#     return parser.parse_args()

# ### Parser Arguments ###
# args = parse_args()

# ### Set Seed ###
# if args.seed is not None:
#     set_seed(args.seed)

# ### Init Accelerator ###
# path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
# accelerator = Accelerator(project_dir=path_to_experiment,
#                           log_with="wandb" if args.log_wandb else None)

# if args.log_wandb:
#     accelerator.init_trackers(args.experiment_name)

# ### Load Model ###
# generator = Generator(args)
# mpd = MultiPeriodDiscriminator()
# msd = MultiScaleDiscriminator()

# # config = HIFIGANConfig(upsample_rates=tuple(args.upsample_rates),
# #                        upsample_kernel_sizes=tuple(args.upsample_kernel_sizes), 
# #                        upsample_initial_channel=args.upsample_initial_channel, 
# #                        residual_block_kernel_sizes=tuple(args.residual_block_kernel_sizes), 
# #                        residual_block_dilation_sizes=args.residual_block_dilation_sizes, 
# #                        mpd_periods=tuple(args.mpd_periods), 
# #                        msd_num_downsamples=args.msd_num_downsamples,
# #                        num_mels=args.num_mels)

# # model = HIFIGAN(config)

# ### Load Optimizers ###
# optim_g = torch.optim.AdamW(generator.parameters(), 
#                             args.learning_rate, 
#                             betas=[args.beta1, args.beta2])

# optim_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), 
#                             args.learning_rate, 
#                             betas=[args.beta1, args.beta2])

# ### Load Schedulers ###
# scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay)
# scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.lr_decay)

# ### Load Dataset ###
# train_dataset = MelDataset(path_to_manifest="train_manifest.txt",
#                            segment_size=args.segment_size, 
#                            n_fft=args.n_fft, 
#                            num_mels=args.num_mels, 
#                            hop_size=args.hop_size, 
#                            window_size=args.window_size, 
#                            sampling_rate=args.sampling_rate, 
#                            fmin=args.fmin, 
#                            fmax=args.fmax, 
#                            fmax_loss=args.fmax_loss, 
#                            n_cache_reuse=args.n_cache_reuse)

# val_dataset = MelDataset(path_to_manifest="val_manifest.txt",
#                          segment_size=args.segment_size, 
#                          n_fft=args.n_fft, 
#                          num_mels=args.num_mels, 
#                          hop_size=args.hop_size, 
#                          window_size=args.window_size, 
#                          sampling_rate=args.sampling_rate, 
#                          fmin=args.fmin, 
#                          fmax=args.fmax, 
#                          fmax_loss=args.fmax_loss, 
#                          n_cache_reuse=args.n_cache_reuse)

# ### Load DataLoader ###
# trainloader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu,
#                          shuffle=True, num_workers=args.num_workers)

# valloader = DataLoader(val_dataset, batch_size=args.batch_size_per_gpu,
#                        shuffle=True, num_workers=args.num_workers)

# ### Prepare Everything ###
# generator, mpd, msd, optim_d, optim_g, trainloader, valloader = accelerator.prepare(
#     generator, mpd, msd, optim_d, optim_g, trainloader, valloader
# )

# ### Training Loop ###
# train = True
# completed_steps = 0 
# log = {"mpd_disc_loss": [], 
#        "msd_disc_loss": [],
#        "mpd_feat_loss": [], 
#        "msd_feat_loss": [],
#        "mpd_gen_loss": [],
#        "msd_gen_loss": [],
#        "train_mel_loss": [], 
#        "val_mel_loss": []}

# pbar = tqdm(range(args.training_iterations), disable=not accelerator.is_main_process)


# while train:

#     for mel, audio, mel_target in trainloader:

#         mel = mel.to(accelerator.device)
#         audio = audio.to(accelerator.device)
#         mel_target = mel_target.to(accelerator.device)
#         audio = audio.unsqueeze(1)

#         ### Generate Waveform from Spectrogram ###
#         # gen_audio = accelerator.unwrap_model(model).generator(mel)
#         gen_audio = generator(mel)
        
#         ### Get Spectrogram of Generated ###
#         gen_audio_spectrogram = compute_mel_spectrogram(audio=gen_audio.squeeze(1), 
#                                                         sampling_rate=args.sampling_rate,
#                                                         n_fft=args.n_fft, 
#                                                         window_size=args.window_size, 
#                                                         hop_size=args.hop_size, 
#                                                         fmin=args.fmin, 
#                                                         fmax=args.fmax_loss,
#                                                         num_mels=args.num_mels)
        
#         ############# DISCRIMINATOR STEP #################
#         optim_d.zero_grad()

#         ### Update MPD Discriminator ###
#         # mpd_real_out, mpd_gen_out, _, _ = accelerator.unwrap_model(model).mpd(audio, gen_audio.detach())
     
#         mpd_real_out, mpd_gen_out, _, _ = mpd(audio, gen_audio.detach())
#         mpd_loss, _, _ = discriminator_loss(mpd_real_out, mpd_gen_out)
#         log["mpd_disc_loss"].append(mpd_loss.item())

#         ### Update MSD Discriminator ###
#         # msd_real_out, msd_gen_out, _, _ = accelerator.unwrap_model(model).msd(audio, gen_audio.detach())
#         msd_real_out, msd_gen_out, _, _ = msd(audio, gen_audio.detach())
#         msd_loss, _, _ = discriminator_loss(msd_real_out, msd_gen_out)
#         log["msd_disc_loss"].append(msd_loss.item())

#         ### Update Discriminator ###
#         total_disc_loss = mpd_loss + msd_loss
#         accelerator.backward(total_disc_loss)
#         optim_d.step()

#         ############# GENERATOR STEP #################
#         optim_g.zero_grad()

#         ### L1 Spectrogram Loss ###
#         loss_mel = F.l1_loss(mel_target, gen_audio_spectrogram)
#         log["train_mel_loss"].append(loss_mel.item())

#         ### Get Discriminator Results ###
#         # mpd_real_out, mpd_gen_out, mpd_real_feats, mpd_gen_feats = accelerator.unwrap_model(model).mpd(audio, gen_audio)
#         # msd_real_out, msd_gen_out, msd_real_feats, msd_gen_feats = accelerator.unwrap_model(model).msd(audio, gen_audio)
#         mpd_real_out, mpd_gen_out, mpd_real_feats, mpd_gen_feats = mpd(audio, gen_audio)
#         msd_real_out, msd_gen_out, msd_real_feats, msd_gen_feats = msd(audio, gen_audio)

#         ### Compute Feature Mapping Loss ###
#         loss_feature_map_mpd = feature_loss(mpd_real_feats, mpd_gen_feats)
#         loss_feature_map_msd = feature_loss(msd_real_feats, msd_gen_feats)
#         log["mpd_feat_loss"].append(loss_feature_map_mpd.item())
#         log["msd_feat_loss"].append(loss_feature_map_msd.item())

#         ### Compute Generator Loss ###
#         loss_gen_mpd, _ = generator_loss(mpd_gen_out) 
#         loss_gen_msd, _ = generator_loss(msd_gen_out)
#         log["mpd_gen_loss"].append(loss_gen_mpd.item())
#         log["msd_gen_loss"].append(loss_gen_msd.item())

#         ### Compute Total Loss ###
#         total_gen_loss = args.lambda_feature_mapping * (loss_feature_map_mpd + loss_feature_map_msd) + \
#             loss_gen_mpd + loss_gen_msd + args.lambda_mel * loss_mel

#         ### Update generator ###
#         accelerator.backward(total_gen_loss)
#         optim_g.step()

#         if completed_steps % args.evaluation_iters == 0:

#             generator.eval()

#             for mel, audio, mel_target in tqdm(valloader, disable=not accelerator.is_main_process):
                
#                 with torch.no_grad():

#                     ### Generate Waveform from Spectrogram ###
#                     gen_audio = generator(mel)

#                     ### Get Spectrogram from Generated Audio ###
#                     gen_audio_spectrogram = compute_mel_spectrogram(audio=gen_audio.squeeze(1), 
#                                                                     sampling_rate=args.sampling_rate,
#                                                                     n_fft=args.n_fft, 
#                                                                     window_size=args.window_size, 
#                                                                     hop_size=args.hop_size, 
#                                                                     fmin=args.fmin, 
#                                                                     fmax=args.fmax_loss,
#                                                                     num_mels=args.num_mels)

#                     val_error = F.l1_loss(mel_target, gen_audio_spectrogram)

#                     log["val_mel_loss"].append(val_error.item())

#             log = {k: np.mean(v) for (k,v) in log.items()}
#             accelerator.log(log, step=completed_steps)

#             if accelerator.is_main_process:
#                 print_log(log, completed_steps)

#             log = {"mpd_disc_loss": [], 
#                    "msd_disc_loss": [],
#                    "mpd_feat_loss": [], 
#                    "msd_feat_loss": [],
#                    "mpd_gen_loss": [],
#                    "msd_gen_loss": [],
#                    "train_mel_loss": [], 
#                    "val_mel_loss": []}

#             generator.train()

#         ### Checkpoint Model (Only need main process for this) ###
#         if (completed_steps % args.checkpoint_iters == 0):
            
#             ### Save Checkpoint ### 
#             path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

#             if accelerator.is_main_process:
#                 pbar.write(f"Saving Checkpoint to {path_to_checkpoint}")

#             ### Make sure that all processes have caught up before saving checkpoint! ###
#             accelerator.wait_for_everyone()

#             ### Save checkpoint using only the main process ###
#             if accelerator.is_main_process:
#                 accelerator.save_state(output_dir=path_to_checkpoint)

#         completed_steps += 1
#         pbar.update(1)

#         if completed_steps >= args.training_iterations:
#             accelerator.print("Completed Training!!!")
#             train = False
#             break
    
#     ### After Epoch is Complete Update Learning Rate ###
#     scheduler_d.step(epoch=completed_steps)
#     scheduler_g.step(epoch=completed_steps)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import MelDataset, compute_mel_spectrogram
from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from loss import feature_loss, generator_loss, discriminator_loss

torch.backends.cudnn.benchmark = True


import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    steps = 0
    state_dict_do = None
    last_epoch = -1

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    trainset = MelDataset("train_manifest.txt", h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, fmax_loss=h.fmax_for_loss)
    

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset("val_manifest.txt", h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
                              fmax_loss=h.fmax_for_loss)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)


    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
    
            y_g_hat_mel = compute_mel_spectrogram(audio=y_g_hat, sampling_rate=h.sampling_rate, n_fft=h.n_fft, window_size=h.win_size, hop_size=h.hop_size, 
                            fmin=h.fmin, fmax=h.fmax_for_loss, num_mels=h.num_mels, center=False, normalized=False).squeeze(1)
            
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

            
                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                          
                            y_g_hat_mel = compute_mel_spectrogram(audio=y_g_hat, sampling_rate=h.sampling_rate, n_fft=h.n_fft, window_size=h.win_size, hop_size=h.hop_size, 
                                                                    fmin=h.fmin, fmax=h.fmax_for_loss, num_mels=h.num_mels, center=False, normalized=False).squeeze(1)

                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

            
                        val_err = val_err_tot / (j+1)
                        print("VALIDATION ERROR: ", val_err)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='train_manifest.txt')
    parser.add_argument('--input_validation_file', default='val_manifest.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()