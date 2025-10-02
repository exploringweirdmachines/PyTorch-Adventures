python -m mytorch.distributed.launch --num_gpus 2 --training_script train_ddp_mnist.py \
    --batch_size 64 \
    --lr 0.001

# python train_ddp_mnist.py \
#     --batch_size 64 \
#     --lr 0.001