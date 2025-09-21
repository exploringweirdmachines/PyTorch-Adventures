python -m mytorch.distributed.launch \
    --num_gpus 2 \
    --training_script train_ddp_mnist.py