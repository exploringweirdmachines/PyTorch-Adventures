python prepare_pretrain_data.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --path_to_data_store "/media/priyam/MASS_NVME_1/mdrive1_data/huggingface/modernbert_large_dataset" \
    --huggingface_cache_dir "/media/priyam/MASS_NVME_1/mdrive1_data/hf_cache" \
    --dataset_split_seed 42 \
    --num_workers 24 \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --large_dataset

# python prepare_pretrain_data.py \
#     --test_split_pct 0.01 \
#     --context_length 512 \
#     --path_to_data_store "/media/priyam/MASS_NVME_1/mdrive1_data/huggingface/distilroberta_gutenberg" \
#     --huggingface_cache_dir "/media/priyam/MASS_NVME_1/mdrive1_data/hf_cache" \
#     --dataset_split_seed 42 \
#     --num_workers 24 \
#     --hf_model_name "distilbert/distilroberta-base" \
#     --batch_size 32 # Reducing batch size because each sample is super long (an entire book!)