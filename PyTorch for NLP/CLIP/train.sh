accelerate launch train.py \
    --working_directory work_dir \
    --experiment_name clip_flicker30k \
    --batch_size 64 \
    --lr 1e-5 \
    --max_steps 5000 \
    --save_steps 2500 \
    --val_steps 500 \
    --log_steps 5 \
    --log_wandb