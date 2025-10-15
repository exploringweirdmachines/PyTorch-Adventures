DATA_PATH="/mnt/datadrive/data/owt_bin"
# 
python -m mytorch.distributed.launch --num_gpus 2 --training_script train_gpt2.py \
    --project_name GPT2Trainer \
    --run_name gpt2-base-owt_fusedCE \
    --working_directory "work_dir" \
    --checkpoint_iter 1000 \
    --context_length 1024 \
    --embed_dim 768 \
    --num_heads 12 \
    --num_blocks 12 \
    --dropout_p 0.0 \
    --mlp_ratio 4 \
    --fused \
    --data_path $DATA_PATH \
    --train_iterations 600000 \
    --warmup_steps 2000 \
    --batch_size 96 \
    --gradient_accumulation 6 \
    --max_lr "6e-4" \
    --min_lr "6e-5" \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --log_iter 5 \
    --mixed_precision \
    --log_wandb