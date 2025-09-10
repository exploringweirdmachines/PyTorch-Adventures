accelerate launch finetune_sft.py \
    --experiment_name "LDM_Pretraining_ft_alpaca" \
    --working_directory "work_dir" \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --path_to_prepped_data "/mnt/datadrive/data/prepped_data/alpaca"