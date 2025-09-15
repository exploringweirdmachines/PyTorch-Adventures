accelerate launch pretrain.py \
    --experiment_name "LDM_Pretraining_large_dataset" \
    --working_directory "work_dir" \
    --hf_model_name "answerdotai/ModernBERT-large" \
    --path_to_prepped_data "/media/priyam/MASS_NVME_1/mdrive1_data/huggingface/modernbert_large_dataset" \
    --num_training_steps 100000 \
    --per_gpu_batch_size 64 \
    --gradient_accumulation_steps 4

# accelerate launch pretrain.py \
#     --experiment_name "distilroberta_ldm_gutenberg" \
#     --working_directory "work_dir" \
#     --hf_model_name "distilbert/distilroberta-base" \
#     --path_to_prepped_data "/media/priyam/MASS_NVME_1/mdrive1_data/huggingface/distilroberta_gutenberg" \
#     --num_training_steps 100000 \
#     --log_wandb