accelerate launch finetune_sft.py \
    --experiment_name "LDM_Pretraining_large_ft_alpaca" \
    --path_to_pretrained_checkpoint "work_dir/LDM_Pretraining_large/checkpoint_400000/model.safetensors" \
    --working_directory "work_dir" \
    --hf_model_name "answerdotai/ModernBERT-large" \
    --path_to_prepped_data "/mnt/datadrive/data/prepped_data/alpaca"

