# import cupy as cp
# from .tensor import Tensor

# #######################
# ### Random sampling ###
# #######################
# def rand(shape, low=0.0, high=1.0, dtype=cp.float32):
#     return Tensor(cp.random.uniform(low, high, size=shape).astype(dtype))

# def randn(shape, mean=0.0, std=1.0, dtype=cp.float32):
#     return Tensor(cp.random.normal(mean, std, size=shape).astype(dtype))

# def randint(shape, low=0, high=10, dtype=cp.int32):
#     return Tensor(cp.random.randint(low, high, size=shape, dtype=dtype))
