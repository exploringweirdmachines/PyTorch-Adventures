# HIFIGAN: Neural Vocoders

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/hifigan_architecture.png?raw=true" alt="drawing" width="600"/>


Many TTS Pipelines have two stages. First generate Mel Spectrograms conditionally on input text, and then second, generate audio from the spectrogram. Unfortunately, traditional signal processing tricks like Griffin Lim don't provide high quality results, leaving you with a robotic sound. So why not train a Neural Network to learn this mapping! There are a bunch of models we can explore here, but for now we will build one of the most popular methods: HifiGAN!

### What is HIFIGAN?
The main idea for HIFIGAN is that when we do Mel spectrograms, each timestep in our Mels coorespond to a specific number of timesteps in the original audio (the downsampling rate of our spectrogram computation). This depends on the setup of our Mel Spectrogram, but in our case, because we use a hop size of 256, each mel timestep cooresponds to 256 audio timesteps. 

If thats the case, why not train a generative model with some transposed convolutions to upsample by a factor of 256! And that is exactly what we do! Of course we need to have some high quality generations, so a few tricks are employed:

1. Mel Loss: We can compute the Mel Spectrogram of the generated audio, and this should look identical to our original Mel spectrogram that we created the audio from!
2. Multi-Scale Discriminator: A discriminator looks at our real audio and our generated audio at different downsampled scaled and tries to discriminate between them
3. Multi-Period Discriminator: A discriminator breaks our audio into stacked periods and then analyzes them, to again discriminate between real and fake audio.

### References
This was mainly based off of the official HIFIGAN implementation you can find here:
- [hifi-gan](https://github.com/jik876/hifi-gan) 

### Prereqs

So to train HIFIGAN, we need a TTS system to generate Mel Spectrograms for! So this will be a continuation of my Tacotron2 Implementation that you can find [here](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20Audio/Tacotron2)


### Pretraining HIFIGAN

The first step will be to train our HIFIGAN on ground truth data. Like before in our Tacotron2 implementation we will be working with the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/). I am reusing the same Train/Test split from my Tacotron2 implementation, you can find the script to produce that [here](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Audio/Tacotron2/prep_splits.py)

Heres how we can pretrain our HIFIGAN! We want to make sure that whatever we are doing here to compute our Mel Spectrograms matches exactly our Mel spectrograms from Tacotron2. Other than that, this should closely match the paper. 

```bash
accelerate launch train.py \
    --experiment_name hifigan \
    --working_directory work_dir \
    --path_to_train_manifest <PATH_TO_TRAIN_METADATA> \
    --path_to_val_manifest <PATH_TO_TEST_METADATA> \
    --training_epochs 3100 \
    --console_out_iters 5 \
    --wandb_log_iters 5 \
    --checkpoint_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.0002 \
    --beta1 0.8 \
    --beta2 0.99 \
    --lr_decay 0.999 \
    --upsample_rates 8 8 2 2 \
    --upsample_kernel_sizes 16 16 4 4 \
    --resblock_kernel_sizes 3 7 11 \
    --resblock_dilation_sizes "((1,3,5),(1,3,5),(1,3,5))" \
    --mpd_periods 2 3 5 7 11 \
    --sampling_rate 22050 \
    --segment_size 8192 \
    --num_mels 80 \
    --n_fft 1024 \
    --window_size 1024 \
    --hop_size 256 \
    --fmin 0 \
    --fmax 8000 \
    --num_workers 16 \
    --log_wandb
```

### Finetuning HIFIGAN

We now have a model that can successfully generate audio from mel spectrograms! But there can be some small issues. Although we trained on Mel Spectrograms that had the same settings as our Tacotron2, our Tacotron2 model could have their own nuances and issues. So we want to finetune this model on actual Tacotron2 mel spectrograms to close any domain discrepancy between ground truth and generations. Our target is still the true audio, we just want to map from generated Tacotron2 spectrograms rather than true Mel Spectrograms. 

#### Generate Mel Spectrograms from Tacotron2

First step is to inference (with teacher forcing) our Mel spectrograms from Tacotron2, and save them as numpy arrays to load later. To do this you can run the following:

```bash
python save_taco_mels.py \
    --path_to_manifest <PATH_TO_TRAIN_METADATA> <PATH_TO_TEST_METADATA>\
    --path_to_save <PATH_TO_SAVE_MELS> \
    --taco_weights <PATH_TO_TACO_WEIGHTS>
```

#### Finetune our HIFIGAN with Saved Spectrograms 

Finally you can finetune your model with our saved spectrograms like the following:

```bash
accelerate launch train.py \
    --experiment_name hifigan_finetune_taco \
    --working_directory work_dir \
    --path_to_train_manifest <PATH_TO_TRAIN_METADATA> \
    --path_to_val_manifest <PATH_TO_TEST_METADATA> \
    --path_to_saved_mels <PATH_TO_SAVE_MELS> \
    --path_to_pretrained_weights <PATH_TO_PRETRAINED_HIFIGAN> \
    --finetune \
    --training_epochs 500 \
    --console_out_iters 5 \
    --wandb_log_iters 5 \
    --checkpoint_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.00005 \
    --beta1 0.8 \
    --beta2 0.99 \
    --lr_decay 0.999 \
    --upsample_rates 8 8 2 2 \
    --upsample_kernel_sizes 16 16 4 4 \
    --resblock_kernel_sizes 3 7 11 \
    --resblock_dilation_sizes "((1,3,5),(1,3,5),(1,3,5))" \
    --mpd_periods 2 3 5 7 11 \
    --sampling_rate 22050 \
    --segment_size 8192 \
    --num_mels 80 \
    --n_fft 1024 \
    --window_size 1024 \
    --hop_size 256 \
    --fmin 0 \
    --fmax 8000 \
    --num_workers 16 \
    --log_wandb 

```

### Results

You can take a look at the ```inference.ipynb``` to see how we can put this all together to generate audio from text! One metric typically used to rate the quality of audio is Mean Opinion Score. This typically requires some participants to listen to the audio and rate it between 1 and 5. I don't have any participants, so instead I am going to use [UTMOS22](https://github.com/sarulab-speech/UTMOS22) which is a model that has been trained to predict MOS of an audio!

| Vocoder | UTMOS |
|----------|----------|
| Griffin Lim    | 1.88|
| HIFIGAN    | 3.37    |
| HIFIGAN + FT  | 3.74 |

We can see that using HIFIGAN gives a meaningfully better score than standard griffin lim! And more importantly, we get a nice boost in score once we finetune our HIFIGAN model on actual Tacotron2 generations as well!
