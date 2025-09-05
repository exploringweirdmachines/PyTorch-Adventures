python train.py \
    --dataset mnist \
    --batch_size 16 \
    --epochs 25 \
    --lr "0.0005" \
    --device cuda:0 \
    --checkpoint_dir work_dir/mnist_chkpts \
    --gens_dir work_dir/mnist_gens \
    --bf16