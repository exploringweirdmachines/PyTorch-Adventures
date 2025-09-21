# CUPYX_DISTRIBUTED_HOST=127.0.0.1 CUPYX_DISTRIBUTED_PORT=13333 \
# python distributed_train_linear_classifier.py --rank 0 --world_size 2 &

# CUPYX_DISTRIBUTED_HOST=127.0.0.1 CUPYX_DISTRIBUTED_PORT=13333 \
# python distributed_train_linear_classifier.py --rank 1 --world_size 2

python -m mytorch.distributed.launch \
    --num_gpus 2 \
    --training_script distributed_train_linear_classifier.py