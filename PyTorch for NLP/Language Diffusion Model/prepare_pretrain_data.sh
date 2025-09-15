python prepare_pretrain_data.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --path_to_data_store "/media/priyam/MASS_NVME_1/mdrive1_data/huggingface/modernbert_large_dataset" \
    --huggingface_cache_dir "/media/priyam/MASS_NVME_1/mdrive1_data/hf_cache" \
    --dataset_split_seed 42 \
    --num_workers 24 \
    --hf_model_name "answerdotai/ModernBERT-large" \
    --large_dataset