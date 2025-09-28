# """
# Our goal here is to homogenize Numpy and Cupy. 
# Cupy is awesome but doenst have CPU support. 
# Numpy is awesome but doesnt have GPU support. 

# Instead of manually toggling between the two, 
# we will create a new object Array, that can 
# sit on the CPU or GPU, and all numpy methods 
# will work on it!
# """

# import numpy as np
# try:
#     import cupy as cp
#     CUDA_AVAILABLE = True
#     NUM_AVAIL_GPUS = cp.cuda.runtime.getDeviceCount()
# except ImportError:
#     cp = None
#     CUDA_AVAILABLE = False
#     NUM_AVAIL_GPUS = 0

# class Array:
#     _binary_ufuncs = {
#         "__add__": "add", "__radd__": "add",
#         "__sub__": "subtract", "__rsub__": "subtract",
#         "__mul__": "multiply", "__rmul__": "multiply",
#         "__truediv__": "true_divide", "__rtruediv__": "true_divide",
#         "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
#         "__matmul__": "matmul", "__rmatmul__": "matmul",
#         "__pow__": "power", "__rpow__": "power",
#         "__mod__": "remainder", "__rmod__": "remainder",
#         "__and__": "bitwise_and", "__rand__": "bitwise_and",
#         "__or__": "bitwise_or", "__ror__": "bitwise_or",
#         "__xor__": "bitwise_xor", "__rxor__": "bitwise_xor",
#         "__lt__": "less", "__le__": "less_equal",
#         "__gt__": "greater", "__ge__": "greater_equal",
#         "__eq__": "equal", "__ne__": "not_equal",
#     }

#     _inplace_ops = {
#         "__iadd__": "add",
#         "__isub__": "subtract",
#         "__imul__": "multiply",
#         "__itruediv__": "true_divide",
#         "__ifloordiv__": "floor_divide",
#         "__imatmul__": "matmul",
#         "__ipow__": "power",
#         "__imod__": "remainder",
#         "__iand__": "bitwise_and",
#         "__ior__": "bitwise_or",
#         "__ixor__": "bitwise_xor",
#     }

#     _unary_ufuncs = {
#         "__neg__": "negative",
#         "__pos__": "positive",
#         "__abs__": "absolute",
#         "__invert__": "invert",
#     }

#     _return_boolean = ("less", "less_equal", "greater", "greater_equal", "equal", "not_equal")

#     def __init__(self, data, device=None, dtype=None):
        
#         ### What device do you want to use? ###
#         if device is not None:

#             ### If a device is provided, no matter where our data is it is moved there ###
#             ### Parse the device string `cpu`, `cuda`, `cuda:{idx}`
#             if device == "cpu":
#                 tgt_device = "cpu"
#                 tgt_device_idx = None

#             elif "cuda" in device:

#                 if not CUDA_AVAILABLE:
#                     raise RuntimeError("CUDA Not supported, check cupy installation")
                
#                 tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=device)

#                 ### Make sure our tgt device is available ###
#                 if tgt_device_idx + 1 > NUM_AVAIL_GPUS:
#                     raise RuntimeError(f"cuda:{tgt_device_idx} does not exist")

#         ### If the device is not provided default to CPU (no matter what data is)
#         ### we can guess based on the device of the input 
#         else:
  
#             if hasattr(data, "device"):
        
#                 if isinstance(data.device, str):
#                     if "cuda" in data.device:
#                         tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=str(data.device))

#                     else:
#                         tgt_device, tgt_device_idx = "cpu", None

#                 elif isinstance(data.device, cp.cuda.device.Device):
#                     tgt_device = "cuda"
#                     tgt_device_idx = data.device.id
#             else:
#                 tgt_device = "cpu"
#                 tgt_device_idx = None

#         ### Figure out dtype, always default to float32 (dtypes stored in .dtypes file) ###
#         if dtype is None:
#             # If data already has a dtype, use it
#             if hasattr(data, "dtype"):
#                 dtype = str(data.dtype)
#                 # If float64, cast down to float32
#                 if dtype == "float64":
#                     dtype = "float32"
#             else:
#                 # Default fallback
#                 dtype = "float32"
#         else:
#             if not isinstance(dtype, str):
#                 raise TypeError("dtypes need to be provided as mytorch.float32 or `float32`")
                
#         ### If input is already Array Type, grab its data
#         if isinstance(data, Array):
#             self._array = data._array

#         ### If its just a np.ndarray or cp.ndarray then just grab that data 
#         elif isinstance(data, (np.ndarray, cp.ndarray)):
#             self._array = data    
        
#         ### If its just an int/float/list/tuple etc... ###
#         else:
#             self._array = np.array(data)

#         ### Map to Correct Device ###
#         self._array = self.__move_array(self._array, 
#                                         src_dev="cpu" if isinstance(self._array, np.ndarray) else "cuda", 
#                                         tgt_dev=tgt_device,
#                                         tgt_dev_idx=tgt_device_idx)
        
#         ### Map to Correct dtype (if on GPU always have to do under context manager) ###
#         ### All cupy ops default on "cuda:0", but if we have tensors in "cuda:1" and are ###
#         ### doing anything to them, we have to make sure we do it under the context ###
#         if "cuda" in self.device and CUDA_AVAILABLE:
#             with cp.cuda.Device(tgt_device_idx):
#                 self._array = self._array.astype(dtype)
#         else:
#             self._array = self._array.astype(dtype)

#     @property
#     def xp(self):
#         """
#         quickly get backend for ops
#         """
#         return np if "cpu" in self.device else cp

#     @property
#     def device(self):
#         """
#         property to always access the device (with id) of the data
#         """
#         if isinstance(self._array, np.ndarray):
#             return "cpu"
#         else:
#             return f"cuda:{self._array.device.id}"

#     @property
#     def dtype(self):
#         """
#         property to always access the dtype of the data
#         """
#         if hasattr(self._array, "dtype"):
#             return str(self._array.dtype)
#         else:
#             raise AttributeError(f"{self} has no dtype")
        
#     def astype(self, dtype):
#         """
#         change the dtype of tensor from outside class
#         """
#         return self._array.astype(dtype)
    
#     def to(self, device):
        
#         ### "cuda" defaults fo first gpu ###
#         if device == "cuda":
#             device = "cuda:0"
        
#         ### if our tgt device is the same as current device, theres nothing to do ###
#         if device == self.device:
#             return self
#         else:
#             if device == "cpu":
#                 tgt_dev = "cpu"
#                 tgt_dev_idx = None
#             else:
#                 tgt_dev, tgt_dev_idx = self.__parse_cuda_str(device)
 
#             return Array(data=self.__move_array(arr=self._array, 
#                                                 src_dev=self.device,
#                                                 tgt_dev=tgt_dev, 
#                                                 tgt_dev_idx=tgt_dev_idx),
#                          device=device, 
#                          dtype=self.dtype)

#     def __parse_cuda_str(self, device_str):
#         tgt_device = "cuda"
#         tgt_device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
#         return tgt_device, tgt_device_idx
    
#     def __move_array(self, arr, src_dev, tgt_dev, tgt_dev_idx=None):
#         if src_dev == tgt_dev:
#             return arr
#         if tgt_dev == "cuda":
#             if not CUDA_AVAILABLE:
#                 raise RuntimeError("CUDA Not supported, check cupy installation")
            
#             ### default device is "cuda:0"
#             if tgt_dev_idx is None:
#                 tgt_dev_idx = 0

#             with cp.cuda.Device(tgt_dev_idx):
#                 return cp.asarray(arr)
#         else:
#             return cp.asnumpy(arr) # cp.asnumpy will also work on numpy arrays, basically a no_op

#     def asnumpy(self):
#         if self.device == "cpu":
#             return self._array
#         return cp.asnumpy(self._array)

#     def _coerce_other(self, other):
#         if isinstance(other, Array):
#             return other._array, other.device
#         if isinstance(other, np.ndarray):
#             return other, "cpu"
#         if CUDA_AVAILABLE and isinstance(other, cp.ndarray):
#             return other, "cuda"
#         return other, None

#     @classmethod
#     def _make_binary_op(cls, ufunc_name, reflect=False):
#         def op(self, other):
#             other_arr, other_dev = self._coerce_other(other)
            
#             if other_dev == "cuda":
#                 other_dev = "cuda:0"

#             if other_dev is not None and other_dev != self.device:
#                 raise RuntimeError(f"Expected all tensors to be on the "
#                  f"same device, but found at least two devices, "
#                  f"{self.device} and {other_dev}!")
            
#             rhs = other_arr
#             xp = self.xp
#             func = getattr(xp, ufunc_name)

#             if reflect:
#                 _in = (rhs, self._array)
#             else:
#                 _in = (self._array, rhs)

#             if xp == cp:
#                 with cp.cuda.Device(self._array.device.id):
#                     res = func(*_in)
#                     if ufunc_name in Array._return_boolean:
#                         res = res.astype(bool)
#             else:
#                 res = func(*_in)
#                 if ufunc_name in Array._return_boolean:
#                     res = res.astype(bool)

#             return Array(res, device=self.device)
        
#         return op

#     @classmethod
#     def _make_unary_op(cls, ufunc_name):
#         def op(self):
#             func = getattr(self.xp, ufunc_name)

#             if self.xp == np:
#                 res = func(self._array)
#             else:
#                 with cp.cuda.Device(self._array.device.id):
#                     res = func(self._array)
#             return Array(res, device=self.device)
#         return op

#     @classmethod
#     def _make_inplace_op(cls, ufunc_name):
#         def op(self, other):
#             other_arr, other_dev = self._coerce_other(other)

#             if other_dev == "cuda":
#                 other_dev = "cuda:0"
                
#             if other_dev is not None and other_dev != self.device:
#                 raise RuntimeError(f"Expected all tensors to be on the "
#                  f"same device, but found at least two devices, "
#                  f"{self.device} and {other_dev}!")
            
#             func = getattr(self.xp, ufunc_name)

#             if self.xp == np:
#                 func(self._array, other_arr, out=self._array)
#             else:
#                 with cp.cuda.Device(self._array.device.id):
#                     func(self._array, other_arr, out=self._array)
#             return self
#         return op

#     def __len__(self):
#         return len(self._array)

#     def __repr__(self):
#         return f"{repr(self._array)}, device={self.device}"

#     def __array_function__(self, func, types, args, kwargs):

#         """
#         For some methods like np.concatenate(), it will trigger the 
#         __array_function__ method instead. Basically, whenever
#         np.concatenate is called with one or more arguments that are 
#         not standard numpy arrays, it goes here to see how to handle them

#         You can find more info here: https://numpy.org/doc/stable/reference/arrays.classes.html

#         """

#         ### I am only building this assuming everything is of Array type ###
#         if not all(issubclass(t, Array) for t in types):
#             return NotImplemented

#         ### Get all devices (should all be the same) ###
#         devices = set()
#         def handler(x):
#             if isinstance(x, Array):
#                 devices.add(x.device)
#                 return x._array
#             elif isinstance(x, (list, tuple)):
#                 return type(x)(handler(y) for y in x)
#             elif isinstance(x, dict):
#                 return {k: handler(v) for k, v in x.items()}
#             else:
#                 return x

#         handled_args = handler(args)
#         handled_kwargs = handler(kwargs)

#         ### If more than one device, we cant do the op ###
#         if len(devices) > 1:
#             raise RuntimeError(f"Expected all tensors to be on the "
#                  f"same device, but found at least two devices, "
#                  f"{device[0]} and {devices[1]}!")

#         ### If nothing we passed in is an array then we will use the ###
#         ### device that we set in self ###
#         if not devices:
#             device = self.device
#         else:
#             device = list(devices)[0]

#         ### Get Backend for Operation ###
#         xp = cp if "cuda" in device else np

#         ### Get the Method ###
#         xp_func = getattr(xp, func.__name__, None)

#         if xp_func is None:
#             return NotImplemented

#         ### Run th Operation ###
#         if "cuda" in device:
#             _, dev_id = self.__parse_cuda_str(device)
#             with cp.cuda.Device(dev_id):
#                 result = xp_func(*handled_args, **handled_kwargs)
#         else:
#             result = xp_func(*handled_args, **handled_kwargs)

#         ### Wrap back into Array ###     
#         return Array(result, device=device)
    
#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         # Extract underlying arrays
#         arrays = [x._array if isinstance(x, Array) else x for x in inputs]
        
#         # Determine device for GPU context
#         device = None
#         for x in inputs:
#             if isinstance(x, Array):
#                 device = x.device
#                 break

#         # Run ufunc
#         if device and "cuda" in device:
#             _, dev_id = self.__parse_cuda_str(device)
#             with cp.cuda.Device(dev_id):
#                 result = getattr(ufunc, method)(*arrays, **kwargs)
#         else:
#             result = getattr(ufunc, method)(*arrays, **kwargs)
        
#         # Wrap result back into Array if it's an array
#         if isinstance(result, (np.ndarray, cp.ndarray)):
#             return Array(result, device=device)
#         return result

#     def __getitem__(self, idx):
        
#         # If idx is an Array itself (boolean mask), convert to underlying array
#         if isinstance(idx, Array):
#             idx = idx._array
        
#         if self.xp == np:
#             result = self._array[idx]

#         else:
#             with cp.cuda.Device(self._array.device.id):
#                 result = self._array[idx]

#         return Array(result, device=self.device)

#     def __setitem__(self, idx, value):
#         # Allow assigning Array, ndarray, or scalar
#         if isinstance(value, Array):
#             value = value._array
#         self._array[idx] = value

#     @classmethod
#     def _wrap_factory(cls, xp_func, *args, device="cpu", dtype="float32", **kwargs):
#         """
#         Wrap numpy/cupy factory functions (zeros, ones, arange, etc.)
#         """
#         # Pick xp from device
#         xp = np if "cpu" in device else cp

#         # Parse CUDA device
#         _, tgt_device_idx = ("cpu", None)
#         if "cuda" in device:
#             _, tgt_device_idx = cls(None).__parse_cuda_str(device)

#         # Run factory function under correct CUDA context
#         if xp == cp:
#             with cp.cuda.Device(tgt_device_idx):
#                 arr = getattr(xp, xp_func)(*args, **kwargs)
#         else:
#             arr = getattr(xp, xp_func)(*args, **kwargs)

#         if dtype is not None:
#             if xp == np:
#                 arr = arr.astype(dtype)
#             else:
#                 with cp.cuda.Device(tgt_device_idx):
#                     arr = arr.astype(dtype)

#         return cls(arr, device=device, dtype=str(arr.dtype))

#     def astype(self, dtype):

#         if self.xp == np:
#             self._array = self._array.astype(dtype)
#         else:
#             _, tgt_dev_idx = self.__parse_cuda_str(self.device)
#             with cp.cuda.Device(tgt_dev_idx):
#                 self._array = self._array.astype(dtype)
        
#         return self

#     def __getattr__(self, name):
#         """
#         To get all the attributes of np or cp that we did not 
#         explicitly define!
#         """
#         if hasattr(self._array, name):
#             return getattr(self._array, name)
#         raise AttributeError(f"'Array' object has no attribute '{name}'")
#         ''
#     @classmethod
#     def zeros(cls, shape, device="cpu", dtype="float32"):
#         return cls._wrap_factory("zeros", shape, device=device, dtype=dtype)

#     @classmethod
#     def ones(cls, shape, device="cpu", dtype="float32"):
#         return cls._wrap_factory("ones", shape, device=device, dtype=dtype)

#     @classmethod
#     def empty(cls, shape, device="cpu", dtype="float32"):
#         return cls._wrap_factory("empty", shape, device=device, dtype=dtype)

#     @classmethod
#     def full(cls, shape, fill_value, device="cpu", dtype="float32"):
#         return cls._wrap_factory("full", shape, fill_value, device=device, dtype=dtype)

#     @classmethod
#     def arange(cls, *args, device="cpu", dtype="float32"):
#         return cls._wrap_factory("arange", *args, device=device, dtype=dtype)

#     @classmethod
#     def eye(cls, N, M=None, k=0, device="cpu", dtype="float32"):
#         return cls._wrap_factory("eye", N, M, k, device=device, dtype=dtype)

#     @classmethod
#     def randn(cls, *args, device="cpu", dtype="float32"):
#         xp = np if "cpu" in device else cp
#         _, tgt_device_idx = ("cpu", None)
#         if "cuda" in device:
#             _, tgt_device_idx = cls(None).__parse_cuda_str(device)
#         if xp == cp:
#             with cp.cuda.Device(tgt_device_idx):
#                 arr = xp.random.randn(*args).astype(dtype)
#         else:
#             arr = xp.random.randn(*args).astype(dtype)
#         return cls(arr, device=device, dtype=str(arr.dtype))

#     @classmethod
#     def tril(cls, x, k=0, device="cpu", dtype="float32"):
#         return cls._wrap_factory("tril", x, k=k, device=device, dtype=dtype)

    
#     @classmethod
#     def zeros_like(cls, other, device="cpu", dtype="float32"):
#         return cls._wrap_factory("zeros_like", other, device=device, dtype=dtype)

#     @classmethod
#     def ones_like(cls, other, device="cpu", dtype="float32"):
#         return cls._wrap_factory("ones_like", other, device=device, dtype=dtype)

# # Attach binary, unary, and inplace operations
# for dunder, ufunc in Array._binary_ufuncs.items():
#     reflect = dunder.startswith("__r")
#     setattr(Array, dunder, Array._make_binary_op(ufunc, reflect=reflect))

# for dunder, ufunc in Array._unary_ufuncs.items():
#     setattr(Array, dunder, Array._make_unary_op(ufunc))

# for dunder, ufunc in Array._inplace_ops.items():
#     setattr(Array, dunder, Array._make_inplace_op(ufunc))

# def test_array_cpu_gpu():
#     devices = ["cpu"]
#     if CUDA_AVAILABLE:
#         for i in range(NUM_AVAIL_GPUS):
#             devices.append(f"cuda:{i}")

#     for device in devices:
#         print(f"\n=== Testing device: {device} ===")
#         a = Array([1, 2, 3, 4], device=device)
#         b = Array([[1, 2], [3, 4]], device=device)

#         # --- Unary ufuncs ---
#         print("abs:", np.abs(a))
#         print("neg:", np.negative(a))
#         print("sqrt:", np.sqrt(a))
#         print("exp:", np.exp(a))
#         print("log:", np.log(a + 1))  # avoid log(0)
        
#         # --- Binary ufuncs ---
#         print("add:", np.add(a, a))
#         print("sub:", np.subtract(a, a))
#         print("mul:", np.multiply(a, a))
#         print("div:", np.divide(a, a))
#         print("pow:", np.power(a, 2))
#         print("mod:", np.mod(a, 2))
#         print("matmul:", np.matmul(b, b))
        
#         # --- Reductions ---
#         print("sum:", np.sum(a))
#         print("mean:", np.mean(a))
#         print("var:", np.var(a))
#         print("std:", np.std(a))
#         print("min:", np.min(a))
#         print("max:", np.max(a))
#         print("argmax:", np.argmax(a))
#         print("argmin:", np.argmin(a))

#         # --- Shape manipulations ---
#         print("reshape:", np.reshape(a, (2, 2)))
#         print("ravel:", np.ravel(b))
#         print("transpose:", np.transpose(b))
#         print("swapaxes:", np.swapaxes(b, 0, 1))
        
#         # --- Stacking / Concatenation ---
#         c = Array([5,6,7,8], device=device)
#         print("concatenate:", np.concatenate([a, c]))
#         print("stack:", np.stack([a, c]))
#         print("hstack:", np.hstack([a, c]))
#         print("vstack:", np.vstack([a, c]))
#         print("column_stack:", np.column_stack([a, c]))

#         # --- Indexing / Slicing ---
#         print("slice:", a[1:])
#         print("fancy indexing:", a[[0,2]])
#         print("boolean indexing:", a[a>2])

#         # --- Random operations ---
#         print("random.rand:", np.random.rand(3,3))
#         print("random.randint:", np.random.randint(0,10,(2,2)))

#         # --- Comparisons ---
#         print("greater:", np.greater(a, 2))
#         print("less_equal:", np.less_equal(a, 3))
#         print("equal:", np.equal(a, 2))

#         # --- Cumulative ---
#         print("cumsum:", np.cumsum(a))
#         print("cumprod:", np.cumprod(a))

#         # --- Copy and astype ---
#         a_copy = np.copy(a)
#         print("copy:", a_copy)

#         a = Array([1, 2, 3], device=device)
#         b = Array([4, 5, 6], device=device)

#         # Direct arithmetic
#         print("a + b =", (a + b).asnumpy())
#         print("a - b =", (a - b).asnumpy())
#         print("a * b =", (a * b).asnumpy())
#         print("a / b =", (a / b).asnumpy())
#         print("a // b =", (a // b).asnumpy())
#         print("a % b =", (a % b).asnumpy())
#         print("a ** 2 =", (a ** 2).asnumpy())

#         # In-place ops
#         c = Array([1, 2, 3], device=device)
#         c += b
#         print("c += b:", c.asnumpy())
#         c -= b
#         print("c -= b:", c.asnumpy())
#         c *= b
#         print("c *= b:", c.asnumpy())
#         c /= b
#         print("c /= b:", c.asnumpy())

#         # Unary
#         print("-a =", (-a).asnumpy())
#         print("+a =", (+a).asnumpy())
#         print("abs(-a) =", abs(-a).asnumpy())

#         # Matmul / @
#         A = Array([[1, 2], [3, 4]], device=device)
#         B = Array([[5, 6], [7, 8]], device=device)
#         print("A @ B =", (A @ B).asnumpy())
#         print("np.matmul(A, B) =", np.matmul(A, B).asnumpy())

#         # NumPy functions
#         print("np.mean(a) =", np.mean(a))
#         print("np.var(a) =", np.var(a))
#         print("np.sum(a) =", np.sum(a))
#         print("np.concatenate([a, b]) =", np.concatenate([a, b]).asnumpy())
#         print("np.transpose(A) =", np.transpose(A).asnumpy())
#         print()

#     print(f"All tests passed for device {device}!")

# if __name__ == "__main__":
#     test_array_cpu_gpu()


"""
Our goal here is to homogenize Numpy and Cupy. 
Cupy is awesome but doesnt have CPU support. 
Numpy is awesome but doesnt have GPU support. 

Instead of manually toggling between the two, 
we will create a new object Array, that can 
sit on the CPU or GPU, and all numpy methods 
will work on it!
"""

import numpy as np
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    NUM_AVAIL_GPUS = cp.cuda.runtime.getDeviceCount()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False
    NUM_AVAIL_GPUS = 0

class Array:
    _binary_ufuncs = {
        "__add__": "add", "__radd__": "add",
        "__sub__": "subtract", "__rsub__": "subtract",
        "__mul__": "multiply", "__rmul__": "multiply",
        "__truediv__": "true_divide", "__rtruediv__": "true_divide",
        "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
        "__matmul__": "matmul", "__rmatmul__": "matmul",
        "__pow__": "power", "__rpow__": "power",
        "__mod__": "remainder", "__rmod__": "remainder",
        "__and__": "bitwise_and", "__rand__": "bitwise_and",
        "__or__": "bitwise_or", "__ror__": "bitwise_or",
        "__xor__": "bitwise_xor", "__rxor__": "bitwise_xor",
        "__lt__": "less", "__le__": "less_equal",
        "__gt__": "greater", "__ge__": "greater_equal",
        "__eq__": "equal", "__ne__": "not_equal",
    }

    _inplace_ops = {
        "__iadd__": "add",
        "__isub__": "subtract",
        "__imul__": "multiply",
        "__itruediv__": "true_divide",
        "__ifloordiv__": "floor_divide",
        "__imatmul__": "matmul",
        "__ipow__": "power",
        "__imod__": "remainder",
        "__iand__": "bitwise_and",
        "__ior__": "bitwise_or",
        "__ixor__": "bitwise_xor",
    }

    _unary_ufuncs = {
        "__neg__": "negative",
        "__pos__": "positive",
        "__abs__": "absolute",
        "__invert__": "invert",
    }

    def __init__(self, data, device=None, dtype=None):
        
        ### What device do you want to use? ###
        if device is not None:

            ### If a device is provided, no matter where our data is it is moved there ###
            ### Parse the device string `cpu`, `cuda`, `cuda:{idx}`
            if device == "cpu":
                tgt_device = "cpu"
                tgt_device_idx = None

            elif "cuda" in device:

                if not CUDA_AVAILABLE:
                    raise RuntimeError("CUDA Not supported, check cupy installation")
                
                tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=device)

                ### Make sure our tgt device is available ###
                if tgt_device_idx + 1 > NUM_AVAIL_GPUS:
                    raise RuntimeError(f"cuda:{tgt_device_idx} does not exist")

        ### If the device is not provided default to CPU (no matter what data is)
        ### we can guess based on the device of the input 
        else:
  
            if hasattr(data, "device"):
        
                if isinstance(data.device, str):
                    if "cuda" in data.device:
                        tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=str(data.device))

                    else:
                        tgt_device, tgt_device_idx = "cpu", None

                elif isinstance(data.device, cp.cuda.device.Device):
                    tgt_device = "cuda"
                    tgt_device_idx = data.device.id
            else:
                tgt_device = "cpu"
                tgt_device_idx = None

        ### Figure out dtype, always default to float32 (dtypes stored in .dtypes file) ###
        if dtype is None:
            # If data already has a dtype, use it
            if hasattr(data, "dtype"):
                current_dtype = str(data.dtype)
                if current_dtype == "float64":
                    dtype = "float32"
                elif current_dtype == "int64":
                    dtype = "int32"
                else:
                    dtype = current_dtype
            else:
                # Default fallback
                dtype = "float32"
        else:
            if not isinstance(dtype, str):
                raise TypeError("dtypes need to be provided as mytorch.float32 or `float32`")
                
        ### If input is already Array Type, grab its data
        if isinstance(data, Array):
            self._array = data._array

        ### If its just a np.ndarray or cp.ndarray then just grab that data 
        elif isinstance(data, (np.ndarray, cp.ndarray)):
            self._array = data    
        
        ### If its just an int/float/list/tuple etc... ###
        else:
            self._array = np.array(data)

        ### Map to Correct Device ###
        src_dev = "cpu" if isinstance(self._array, np.ndarray) else f"cuda:{self._array.device.id}"
        self._array = self.__move_array(self._array, 
                                        src_dev=src_dev, 
                                        tgt_dev=tgt_device,
                                        tgt_dev_idx=tgt_device_idx)
        
        ### Map to Correct dtype (if on GPU always have to do under context manager) ###
        ### All cupy ops default on "cuda:0", but if we have tensors in "cuda:1" and are ###
        ### doing anything to them, we have to make sure we do it under the context ###
        current_dtype = str(self._array.dtype)
        if current_dtype != dtype:
            if "cuda" in tgt_device and CUDA_AVAILABLE:
                with cp.cuda.Device(tgt_device_idx):
                    self._array = self._array.astype(dtype)
            else:
                self._array = self._array.astype(dtype)

        ### Cache frequently accessed attributes ###
        self._xp = np if isinstance(self._array, np.ndarray) else cp
        self._dev_id = None if self._xp is np else self._array.device.id
        self._device = "cpu" if self._xp is np else f"cuda:{self._dev_id}"
        self._dtype = str(self._array.dtype)

    @property
    def xp(self):
        return self._xp

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def size(self):
        return self._array.size

    @property
    def T(self):
        return Array(self._array.T, device=self._device)

    def astype(self, dtype):
        if self._dtype == dtype:
            return self
        if self._xp is np:
            self._array = self._array.astype(dtype)
        else:
            with cp.cuda.Device(self._dev_id):
                self._array = self._array.astype(dtype)
        self._dtype = str(self._array.dtype)
        return self
    
    def to(self, device):
        
        ### "cuda" defaults to first gpu ###
        if device == "cuda":
            device = "cuda:0"
        
        ### if our tgt device is the same as current device, theres nothing to do ###
        if device == self._device:
            return self
        else:
            if device == "cpu":
                tgt_dev = "cpu"
                tgt_dev_idx = None
            else:
                tgt_dev, tgt_dev_idx = self.__parse_cuda_str(device)
 
            return Array(data=self.__move_array(arr=self._array, 
                                                src_dev=self._device,
                                                tgt_dev=tgt_dev, 
                                                tgt_dev_idx=tgt_dev_idx),
                         device=device, 
                         dtype=self._dtype)

    def __parse_cuda_str(self, device_str):
        tgt_device = "cuda"
        tgt_device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
        return tgt_device, tgt_device_idx
    
    def __move_array(self, arr, src_dev, tgt_dev, tgt_dev_idx=None):
        src_tgt = src_dev if src_dev == "cpu" else "cuda"
        src_idx = None if src_dev == "cpu" else int(src_dev.split(":")[-1])
        tgt_idx = tgt_dev_idx if tgt_dev == "cuda" else None
        if src_tgt == tgt_dev and src_idx == tgt_idx:
            return arr
        if tgt_dev == "cuda":
            if not CUDA_AVAILABLE:
                raise RuntimeError("CUDA Not supported, check cupy installation")
            
            ### default device is "cuda:0"
            if tgt_dev_idx is None:
                tgt_dev_idx = 0

            with cp.cuda.Device(tgt_dev_idx):
                return cp.asarray(arr)
        else:
            return cp.asnumpy(arr) # cp.asnumpy will also work on numpy arrays, basically a no_op

    def asnumpy(self):
        if self._device == "cpu":
            return self._array
        return cp.asnumpy(self._array)

    def _coerce_other(self, other):
        if isinstance(other, Array):
            return other._array, other._device
        if isinstance(other, np.ndarray):
            return other, "cpu"
        if CUDA_AVAILABLE and isinstance(other, cp.ndarray):
            return other, f"cuda:{other.device.id}"
        return other, None

    @classmethod
    def _make_binary_op(cls, ufunc_name, reflect=False):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            rhs = other_arr
            xp = self._xp
            func = getattr(xp, ufunc_name)

            if reflect:
                _in = (rhs, self._array)
            else:
                _in = (self._array, rhs)

            if xp is cp:
                with cp.cuda.Device(self._dev_id):
                    res = func(*_in)
            else:
                res = func(*_in)

            return Array(res, device=self._device)
        
        return op

    @classmethod
    def _make_unary_op(cls, ufunc_name):
        def op(self):
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                res = func(self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    res = func(self._array)
            return Array(res, device=self._device)
        return op

    @classmethod
    def _make_inplace_op(cls, ufunc_name):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            # print(other_dev, self._device)
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                func(self._array, other_arr, out=self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    func(self._array, other_arr, out=self._array)
            return self
        return op

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return f"{repr(self._array)}, device={self._device}"

    def __array_function__(self, func, types, args, kwargs):

        """
        For some methods like np.concatenate(), it will trigger the 
        __array_function__ method instead. Basically, whenever
        np.concatenate is called with one or more arguments that are 
        not standard numpy arrays, it goes here to see how to handle them

        You can find more info here: https://numpy.org/doc/stable/reference/arrays.classes.html

        """

        ### I am only building this assuming everything is of Array type ###
        if not all(issubclass(t, Array) for t in types):
            return NotImplemented

        ### Get all devices (should all be the same) ###
        devices = set()
        def handler(x):
            if isinstance(x, Array):
                devices.add(x._device)
                return x._array
            elif isinstance(x, (list, tuple)):
                return type(x)(handler(y) for y in x)
            elif isinstance(x, dict):
                return {k: handler(v) for k, v in x.items()}
            else:
                return x

        handled_args = handler(args)
        handled_kwargs = handler(kwargs)

        ### If more than one device, we cant do the op ###
        if len(devices) > 1:
            raise RuntimeError(f"Expected all tensors to be on the "
                f"same device, but found at least two devices!")

        ### If nothing we passed in is an array then we will use the ###
        ### device that we set in self ###
        if not devices:
            device = self._device
        else:
            device = list(devices)[0]

        ### Get Backend for Operation ###
        xp = cp if "cuda" in device else np

        ### Get the Method ###
        xp_func = getattr(xp, func.__name__, None)

        if xp_func is None:
            return NotImplemented

        ### Run the Operation ###
        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = xp_func(*handled_args, **handled_kwargs)
        else:
            result = xp_func(*handled_args, **handled_kwargs)

        ### Wrap back into Array if result is an array ###
        if isinstance(result, (np.ndarray, cp.ndarray if CUDA_AVAILABLE else type(None))):
            return Array(result, device=device)
        return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Extract underlying arrays
        arrays = [x._array if isinstance(x, Array) else x for x in inputs]
        
        # Determine device for GPU context
        device = None
        for x in inputs:
            if isinstance(x, Array):
                device = x._device
                break

        # Run ufunc
        if device and "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = getattr(ufunc, method)(*arrays, **kwargs)
        else:
            result = getattr(ufunc, method)(*arrays, **kwargs)
        
        # Wrap result back into Array if it's an array
        if isinstance(result, (np.ndarray, cp.ndarray)):
            return Array(result, device=device)
        return result

    # def __getitem__(self, idx):
        
    #     # If idx is an Array itself (boolean mask), convert to underlying array
    #     if isinstance(idx, Array):
    #         idx = idx._array
        
    #     if self._xp is np:
    #         result = self._array[idx]

    #     else:
    #         with cp.cuda.Device(self._dev_id):
    #             result = self._array[idx]

    #     return Array(result, device=self._device)

    def __getitem__(self, idx):
    
        def _coerce_index(index):
            if isinstance(index, tuple):
                return tuple(_coerce_index(i) for i in index)
            if isinstance(index, Array):
                return index._array
            # Handle mytorch.Tensor (assumes it has .data as Array)
            if hasattr(index, 'data') and isinstance(index.data, Array):
                return index.data._array
            return index
        
        idx = _coerce_index(idx)
      
        if self.xp == np:
            result = self._array[idx]
        else:
            with cp.cuda.Device(self._array.device.id):
                result = self._array[idx]

        return Array(result, device=self.device)

    def __setitem__(self, idx, value):
        # Allow assigning Array, ndarray, or scalar
        if isinstance(value, Array):
            value = value._array
        self._array[idx] = value

    @classmethod
    def _wrap_factory(cls, xp_func, *args, device="cpu", dtype="float32", **kwargs):
        """
        Wrap numpy/cupy factory functions (zeros, ones, arange, etc.)
        """
        # Pick xp from device
        xp = np if "cpu" in device else cp

        # Parse CUDA device
        _, tgt_device_idx = ("cpu", None)
        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Run factory function under correct CUDA context
        if xp == cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = getattr(xp, xp_func)(*args, **kwargs)
        else:
            arr = getattr(xp, xp_func)(*args, **kwargs)

        if dtype is not None:
            current_dtype = str(arr.dtype)
            if current_dtype != dtype:
                if xp == np:
                    arr = arr.astype(dtype)
                else:
                    with cp.cuda.Device(tgt_device_idx):
                        arr = arr.astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    def __getattr__(self, name):
        """
        To get all the attributes of np or cp that we did not 
        explicitly define!
        """
        if hasattr(self._array, name):
            return getattr(self._array, name)
        raise AttributeError(f"'Array' object has no attribute '{name}'")
        
    @classmethod
    def zeros(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("zeros", shape, device=device, dtype=dtype)

    @classmethod
    def ones(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("ones", shape, device=device, dtype=dtype)

    @classmethod
    def empty(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("empty", shape, device=device, dtype=dtype)

    @classmethod
    def full(cls, shape, fill_value, device="cpu", dtype="float32"):
        return cls._wrap_factory("full", shape, fill_value, device=device, dtype=dtype)

    @classmethod
    def arange(cls, *args, device="cpu", dtype="float32"):
        return cls._wrap_factory("arange", *args, device=device, dtype=dtype)

    @classmethod
    def eye(cls, N, M=None, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("eye", N, M, k, device=device, dtype=dtype)

    @classmethod
    def randn(cls, *args, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        _, tgt_device_idx = ("cpu", None)
        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)
        if xp == cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.randn(*args).astype(dtype)
        else:
            arr = xp.random.randn(*args).astype(dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def tril(cls, x, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("tril", x, k=k, device=device, dtype=dtype)

    
    @classmethod
    def zeros_like(cls, other, device="cpu", dtype="float32"):
        return cls._wrap_factory("zeros_like", other, device=device, dtype=dtype)

    @classmethod
    def ones_like(cls, other, device="cpu", dtype="float32"):
        return cls._wrap_factory("ones_like", other, device=device, dtype=dtype)

# Attach binary, unary, and inplace operations
for dunder, ufunc in Array._binary_ufuncs.items():
    reflect = dunder.startswith("__r")
    setattr(Array, dunder, Array._make_binary_op(ufunc, reflect=reflect))

for dunder, ufunc in Array._unary_ufuncs.items():
    setattr(Array, dunder, Array._make_unary_op(ufunc))

for dunder, ufunc in Array._inplace_ops.items():
    setattr(Array, dunder, Array._make_inplace_op(ufunc))