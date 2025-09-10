python prepare_sft_data.py \
    --test_split_pct 0.01 \
    --context_length 1024 \
    --path_to_data_store "/mnt/datadrive/data/prepped_data/alpaca" \
    --huggingface_cache_dir "/mnt/datadrive/data/huggingface_cache" \
    --dataset_split_seed 42 \
    --num_workers 24 \
    --hf_model_name "answerdotai/ModernBERT-base"