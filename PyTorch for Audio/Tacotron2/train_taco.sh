# accelerate launch train_taco.py \
#     --experiment_name tacotron2 \
#     --run_name no_wd \
#     --working_directory work_dir/no_wd \
#     --save_audio_gen work_dir/save_gen_no_wd \
#     --path_to_train_manifest data/train_metadata.csv \
#     --path_to_val_manifest data/test_metadata.csv \
#     --training_epochs 50 \
#     --console_out_iters 5 \
#     --wandb_log_iters 5 \
#     --checkpoint_epochs 25 \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --weight_decay 0 \
#     --adam_eps "1e-6" \
#     --min_learning_rate "1e-5" \
#     --character_embed_dim 512 \
#     --encoder_kernel_size 5 \
#     --encoder_n_convolutions 3 \
#     --encoder_embed_dim 512 \
#     --encoder_dropout_p 0.5 \
#     --decoder_rnn_embed_dim 1024 \
#     --decoder_dropout_p 0.1 \
#     --decoder_prenet_dim 256 \
#     --decoder_prenet_depth 2 \
#     --decoder_prenet_dropout_p 0.5 \
#     --decoder_postnet_num_convs 5 \
#     --decoder_postnet_n_filters 512 \
#     --decoder_postnet_kernel_size 5 \
#     --decoder_postnet_dropout_p 0.5 \
#     --attention_dim 128 \
#     --attention_dropout_p 0.1 \
#     --attention_location_n_filters 32 \
#     --attention_location_kernel_size 31 \
#     --sampling_rate 22050 \
#     --num_mels 80 \
#     --n_fft 1024 \
#     --window_size 1024 \
#     --hop_size 256 \
#     --min_db "-100" \
#     --max_scaled_abs 4 \
#     --fmin 0 \
#     --fmax 8000 \
#     --num_workers 32 \
#     --log_wandb

accelerate launch train_taco.py \
    --experiment_name tacotron2 \
    --run_name w_wd \
    --working_directory work_dir/w_wd \
    --save_audio_gen work_dir/save_gen_w_wd \
    --path_to_train_manifest data/train_metadata.csv \
    --path_to_val_manifest data/test_metadata.csv \
    --training_epochs 50 \
    --console_out_iters 5 \
    --wandb_log_iters 5 \
    --checkpoint_epochs 25 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay "1e-6" \
    --adam_eps "1e-6" \
    --min_learning_rate "1e-5" \
    --character_embed_dim 512 \
    --encoder_kernel_size 5 \
    --encoder_n_convolutions 3 \
    --encoder_embed_dim 512 \
    --encoder_dropout_p 0.5 \
    --decoder_rnn_embed_dim 1024 \
    --decoder_dropout_p 0.1 \
    --decoder_prenet_dim 256 \
    --decoder_prenet_depth 2 \
    --decoder_prenet_dropout_p 0.5 \
    --decoder_postnet_num_convs 5 \
    --decoder_postnet_n_filters 512 \
    --decoder_postnet_kernel_size 5 \
    --decoder_postnet_dropout_p 0.5 \
    --attention_dim 128 \
    --attention_dropout_p 0.1 \
    --attention_location_n_filters 32 \
    --attention_location_kernel_size 31 \
    --sampling_rate 22050 \
    --num_mels 80 \
    --n_fft 1024 \
    --window_size 1024 \
    --hop_size 256 \
    --min_db "-100" \
    --max_scaled_abs 4 \
    --fmin 0 \
    --fmax 8000 \
    --num_workers 32 \
    --log_wandb