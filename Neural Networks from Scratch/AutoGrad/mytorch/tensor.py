import os
import cupy as cp
import weakref
from contextlib import contextmanager
import gc

##### QUICK ENABLE FOR TENSOR CORE OPS ###
device = cp.cuda.Device()
# string containing the major index and the minor index. 
# For example, compute capability 3.5 is represented by the string ‘35’.
cc_major, cc_minor = device.compute_capability 
if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"
##########################################
    
def cupy_prod(x, axis=None, dtype=None, keepdims=False):
    """
    Quick helper method as cp.prod(x) doesnt work if x is a list
    unlike numpy that internally converts to numpy array before 
    doing the prod
    """
    if not isinstance(x, cp.ndarray):
        x = cp.array(x)
    return cp.prod(x, axis=axis, dtype=dtype, keepdims=keepdims).astype(cp.float32)

@contextmanager
def no_grad():
    old_state = Tensor._build_graph
    Tensor._build_graph = False
    try:
        # Yield is where we get the 'with' statement to execute
        yield
    finally:
        # Finally ensures that no matter what the original _build_graph is restored
        Tensor._build_graph = old_state

class Tensor:

    ### For Context Manager no_grad() ###
    _build_graph = True

    def __init__(self, 
                 data, 
                 requires_grad=False, 
                 grad_fn=None,
                 grad_fn_name=None):
        
        ### Store Passed in Variables ###
        self.data = self._toarray(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad_fn_name = grad_fn_name
        self.shape = self.data.shape

        ### Container to Store Children (output) of Every Operation ###
        ### (i.e. if we add tensor A and B to create C, C is the child of A and child of B) ###
        self._parents = ()

        self.is_leaf = self.requires_grad and (grad_fn is None)
        
        ### If we actually want to store gradient, Initialize with Zeros ###
        if self.requires_grad:
            self.grad = None

    def __repr__(self):
        if self.grad_fn_name is None:
            if self.requires_grad:
                return f"{self.data}, requires_grad={self.requires_grad}, device=cuda:{self.data.device.id}"
            else:
                return f"{self.data}, device=cuda:{self.data.device.id}"
        else:
            return f"{self.data}, grad_fn={self.grad_fn_name}, device=cuda:{self.data.device.id}"
    
    @classmethod
    def build_graph_enabled(cls):
        return cls._build_graph
    
    def _check_broadcast(self, a, b):

        ## Verify that two numpy arrays are broadcastable ###
        ## This means a and b have the same number of dimensions ###
        ## I.E (1x3) + (1x1) summation is broadcasting

        ### We only really care about this when both a and b requires gradients ###
        ### as if they dont, then either a or b are just some constant ###

        ## Numpy technically supports broadcasting even when the dimensionality ###
        ## is not the same (1 x 3) + (1, ) but we wont for simplicity! ###
        if (len(a.shape) != len(b.shape)) and (a.requires_grad and b.requires_grad):
            raise ValueError(f"Incompatible Operation between {a.shape} and {b.shape}")
        
    def _broadcasted_grad_accumulate(self, x_shape, x_grad):
        
        ### This function is crucial and taken from https://github.com/eduardoleao052/Autograd-from-scratch ###
        ### Much of our convenient operations are broadcasting! For example, we can add a tensor of size (A x B)
        ### to another tensor (A x B x C). What broadcasting does is automatically add our (A x B) tensor to C number of 
        ### the (A x B) tensors found in our larger tensor. 

        # (A x B) + (A x B x C) -> (A x B x C)

        ### How does this actually happen? By repeating! Basically our smaller (A x B) tensor is repeated C times to create a 
        ### C * (A x B) which creates an (A x B x C) tensor, and then added to the second (A x B x C) tensor. 

        ### In Neural Networks, a typical dimension we broadcast over is the Batch Dimension. If we have N samples in our neural network ###
        ### we technically can pass in every sample in N one at a time to our neural network. This is obviously very inefficient though ###
        ### and so we broadcast all the components of the neural network over the batch dimension so we can parallize the computation in fast CUDA code ###

        ### Now the problem we have. We want to compute the gradients to update our network based on a batch of samples. Again, technically we could ###
        ### pass in a single sample at a time and then add up (accumulate) the gradients for each sample. But this is a bit slow! So instead we need to just ###
        ### do it all in batches which will cause some hassles. 

        ### Lets say we are solving y = mx + b
        ### b is a (1, 1) tensor, i.e. really just a single value
        ### w is a (1, 1) tensor, i.e. really just a single value
        ### x is a (N x 1) tensor, so we have N samples in our batch and each sample has one feature
        ### y is a (N x 1) tensor, so each input x has an output y with one feature (the thing we are predicting)

        ### Ater we compute y we typically compute our loss, lets say mean squared error (sum(y - y_hat)**2)/N
        ### To learn our coefficient and intercept we need to compute dL/dW and dL/db. Because of chain rule this becomes:

        ### dL/dW = dL/dY * dY/dW
        ### dL/db = dL/dY * dY/db

        ### So this term: dL/dY. For which Y are we doing this? In a batch we have N outputs. Well, we are doing it for all of them, 
        ### and summing them together! (as the operation for the mean loss has a sum inside it). So in the same way, we need to create a 
        ### vector of loss values such as [dL/dY_0, dL/dY_1, dL/dY_2, ... dL/dY_N]. And then this (Nx1) gradient tensor of dL/dY
        ### goes onto the next step of backprop. 

        ### To update w, we saw before that the formula is dL/dW = x^T @ dL/dY. This will be a (1 x N) tensor multiplied by a (N x 1) tensor outputing
        ### a single (1,1) tensor that we need to update our gradients. This "sum" operation across the batch is thus built right into a matrix 
        ### multiplication

        ### To update b on the other hand, the formula was dL/db = sum(dL/dY_i). Thus we need to manually do the sum here!
        ### Therefore we need to accumulate any extra dimensions we have! Our gradient vector for dL/dY is (N x 1), but the 
        ### actual tensor for our single bias value is (1,1). Therefore, on the first dimension we have a discrepancy, 
        ### we have a 1 in our tensor, but N in the gradient, so we must add across the dimension and create an accumulated (1,1) gradient 

        grad_shape = x_grad.shape

        assert len(x_shape) == len(grad_shape), "Gradient and tensor shapes must be the same length! Only different by broadcasting"

        # for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)):
        #     ### If our tensor dim is 1 but the grad dim is not, accumulate on that dimension ###
        #     if (x_dim == 1) and (grad_dim != 1):
        #         x_grad = x_grad.sum(axis=idx, keepdims=True)
        #     ### Otherwise verify that our x_dim and grad dim are the same!! 
        #     else:
        #         assert (x_dim == grad_dim)

        ### Now instead of checking every dim and adding, just add all at once to vectorize! ###
        sum_axes = [idx for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)) if x_dim == 1 and grad_dim != 1]
        if sum_axes:
            x_grad = cp.sum(x_grad, axis=tuple(sum_axes), keepdims=True)

        return x_grad
    
    def backward(tensor, grad=None):
       
        # Initialize output gradient
        if grad is None:
            grad = cp.ones_like(tensor.data, dtype=cp.float32)
        tensor.grad = grad

        # Build topo-order
        visited = set()
        topo_order = []

        def build_topo(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            parents = getattr(t, "_parents", ())
            if parents is None:
                parents = ()
            for parent_ref in parents:
                parent = parent_ref()
                if parent is not None:
                    build_topo(parent)
            topo_order.append(t)

        build_topo(tensor)

        # Iterate in reverse topological order
        for t in reversed(topo_order):
            if t.grad_fn is not None:
                t.grad_fn(t.grad)  # accumulate into parents

                # Clear backward references so they can be GC'ed
                t.grad_fn = None
                t._parents = None

    def __add__(self, val):

        """
        Sum of two tensors (with accumulation for brodcasting)
        O = A + B
        dO/dA = 1
        dO/dB = 1
        """

        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)
        
        ### Check Broadcast ###
        self._check_broadcast(self, val)

        ### Use Numpy __add__ to actually add tensors together ###
        output = self.data + val.data
        
        ### Define Backward Function ###
        def _add_backward(input_grad):
            if self.requires_grad:
                self_grad = self._broadcasted_grad_accumulate(self.shape, input_grad)

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

            if val.requires_grad:
                val_grad = self._broadcasted_grad_accumulate(val.shape, input_grad)
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_add_backward if requires_grad else None,
                        grad_fn_name="<AddBackward>" if requires_grad else None)

        # set parents
        if requires_grad:
            output._add_parents(self, val)

        return output
    
    def __radd__(self, val):

        """
        add is not an ordered operation, A + B is the same as B + A

        In A + B, our self is A and val is B
        When we do A + B, what is really happening is A.__add__(B). 

        But if A is an integer and B is a Tensor, python integers dont know how to work with our
        own tensor operations. This will throw an error and then try __radd__.  
    
        __radd__ will reverse the operands and do B.__add__(A), using our own Tensor __add__ written above instead.  
        Our __add__ we wrote for the tensor does know how to interface python numbers and tensors so we can then do the operation!

        """
        return self + val

    def __sub__(self, val):

        """
        Same as __add__ but now subtraction (with accumulation for broadcasting)
        O = A - B
        dO/dA = 1
        dO/dB = -1
        """
    
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val)
        
        ### Check Broadcast ###
        self._check_broadcast(self, val)

        ### Use Numpy __add__ to actually add tensors together ###
        output = self.data - val.data
        
        ### Define Backward Function ###
        def _sub_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
            if val.requires_grad:
                val_grad = -input_grad
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

        ### Wrap our output in our tensor object ###
        ### We will compute grad on this if self or val are also tracking gradients ###
        ### We also store the backward_fn for the backward pass ###
        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sub_backward if requires_grad else None,
                        grad_fn_name="<SubBackward>" if requires_grad else None)#.astype(cp.float32)
        
        ### This output is the child of the inputs a and b ###
        if requires_grad:
            output._add_parents(self, val)
        
        return output
    
    def __rsub__(self, val):

        """
        Subtraction is an ordered operation. Lets say we want A - B where A is self and B is val
        if A is not a tensor (i.e. an int or float), __sub__ will throw an error as it doesnt know
        how to do an operation with our own tensor.

        This will enter __rsub__ where we flip the operands where B is now self and A is val. If we want
        A - B, we need to do -1 * B + A, using our __add__. 

        There are a bunch of ways to handle these exceptions, this is just one of them!
        """

        return -1 * self + val

    def __mul__(self, val):

        """
        Element-wise multiplication of two tensors (with accumulation for broadcasting)

        O = A * B
        dO/dA = B
        do/dB = A
        """

        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val).astype(cp.float32)

        ### Check Broadcast ### 
        self._check_broadcast(self, val)
            
        output = self.data * val.data

        def _mul_backward(input_grad):
            
            if self.requires_grad:
                self_grad = input_grad * val.data
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
            
            if val.requires_grad:
                val_grad = input_grad * self.data
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad


        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_mul_backward if requires_grad else None,
                        grad_fn_name="<MulBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            output._add_parents(self, val)

        return output
    
    def __rmul__(self, val):
        return self * val

    def __neg__(self):
        return self * cp.array(-1.0, dtype=self.data.dtype)

    def __matmul__(self, val):
        if not isinstance(val, Tensor):
            val = Tensor(val)
        
        ### Preallocate Memory for MatMul ###
        non_matmul_shapes = self.shape[:-2]
        output_shape = (*non_matmul_shapes, self.data.shape[-2], val.data.shape[-1])
        prealloc = cp.empty(output_shape, dtype=self.data.dtype)

        ### Compute MatMul ###
        output_data = cp.matmul(self.data, val.data, out=prealloc)

        def _matmul_backward(input_grad):

            if self.requires_grad:
                prealloc_grad_self = cp.empty(shape=self.shape, dtype=self.data.dtype)
                grad_self = cp.matmul(input_grad, val.data.swapaxes(-1, -2), out=prealloc_grad_self)
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

            if val.requires_grad:
                prealloc_grad_val = cp.empty(shape=val.shape, dtype=self.data.dtype)
                grad_val = cp.matmul(self.data.swapaxes(-1, -2), input_grad, out=prealloc_grad_val)
                
                if val.grad is None:
                    val.grad = grad_val
                else:
                    val.grad += grad_val


        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        out = Tensor(
            output_data,
            requires_grad=requires_grad,
            grad_fn=_matmul_backward if requires_grad else None,
            grad_fn_name="<MatmulBackward>" if requires_grad else None
        )#.astype(cp.float32) 

        if requires_grad:
            out._add_parents(self, val)

        return out
        
    def __truediv__(self, val):

        """
        Element-wise Division of two tensors (accumulated grad for broadcasting)

        O = A/B
        dO/dA = 1/B
        dO/dB = -A/B^2

        """

        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val).astype(cp.float32)

        ### Check Broadcast ###
        self._check_broadcast(self, val)

        output = self.data / val.data
  
        def _div_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad / val.data
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

            if val.requires_grad:
                val_grad = input_grad * -1 * self.data / (val.data**2)
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad
        
        ### Convert to Tensor ###
        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_div_backward if requires_grad else None,
                        grad_fn_name="<DivBackward>" if requires_grad else None)#.astype(cp.float32) 

        ### This output is the child of the inputs a and b ###
        if requires_grad:
            output._add_parents(self, val)

        return output

    def __rtruediv__(self, val):
        
        """
        Div is an ordered operation. Lets say we want A/B, in the case of __div__ A is self and B is val. 
        if A is not a Tensor (i.e. an int or float), A / B will throw an error beacuse we only can divide a tensor by a tensor
        In this case, __rtruediv__ will be called where A is now val and B is self (the operands have been flipped)
        We can then convert A (our non-tensor) which is in val to a tensor and then perform val / self to call __div__ again where
        A and B are both now tensors
        """
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val).astype(cp.float32)

        return val / self

    def __pow__(self, exponent):

        """
        Element-wise exponentiation of matrix (assuming exponent is non-learnable for simplicity)
        O = A^K
        dO/dA = K * A^(k-1)
        """

        output = self.data ** exponent
    
        def _pow_backward(input_grad):
            self_grad = input_grad * (exponent * self.data ** (exponent-1))
            
            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_pow_backward if requires_grad else None,
                        grad_fn_name="<PowBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            output._add_parents(self)

        return output
    
    def __getitem__(self, idx):
        """
        Supports slices, ints, arrays, and tuple-of-arrays indexing.
        """

        # Convert Tensor indices to cp arrays
        if isinstance(idx, Tensor):
            idx = idx.data

        if isinstance(idx, (list, tuple)):
            idx = tuple(self._toarray(i) for i in idx)

        # Forward: use standard NumPy fancy indexing
        out_data = self.data[idx]

        def _index_backward(input_grad):

            if self.requires_grad:
                if self.grad is None:
                    self.grad = cp.zeros_like(self.data, dtype=cp.float32)

                # Elementwise assignment for fancy indexing
                self.grad[idx] += input_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_index_backward if requires_grad else None,
                    grad_fn_name="<IndexBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            out._add_parents(self)

        return out
    
    def __eq__(self, other): # ==
        return Tensor(self.data == (other.data if isinstance(other, Tensor) else other), requires_grad=False)

    def __ne__(self, other):  # !=
        return ~(self == other)

    def __lt__(self, other):  # <
        return Tensor(self.data < (other.data if isinstance(other, Tensor) else other), requires_grad=False)

    def __le__(self, other):  # <=
        return Tensor(self.data <= (other.data if isinstance(other, Tensor) else other), requires_grad=False)

    def __gt__(self, other):  # >
        return Tensor(self.data > (other.data if isinstance(other, Tensor) else other), requires_grad=False)

    def __ge__(self, other):  # >=
        return Tensor(self.data >= (other.data if isinstance(other, Tensor) else other), requires_grad=False)

    def __len__(self):
        return self.shape[0]
    
    def transpose(self, dim1, dim2):
        """
        Swap two dimensions of the tensor.
        """
        out_data = self.data.swapaxes(dim1, dim2)
 
        def _transpose_backward(input_grad):
            # Just swap back the same two dims
            if self.requires_grad:
                self_grad = input_grad.swapaxes(dim1, dim2)

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_transpose_backward if requires_grad else None,
                    grad_fn_name="<TransposeBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            out._add_parents(self)

        return out
    
    def permute(self, *dims):
        """
        Permute tensor dimensions according to dims.
        Example: (0, 2, 1) will reorder axes in that order.
        """
        out_data = cp.transpose(self.data, axes=dims)

        def _permute_backward(input_grad):
            if self.requires_grad:
                # Inverse permutation
                inv_dims = cp.argsort(dims)
                self_grad = cp.transpose(input_grad, axes=inv_dims)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_permute_backward if requires_grad else None,
                    grad_fn_name="<PermuteBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            out._add_parents(self)

        return out

    def exp(self):
        """
        Element-wise exponentiation of the base e.
        O = e^A
        dO/dA = e^A
        """
        out_data = cp.exp(self.data)

        def _exp_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad * out_data  # use forward output to save recomputation
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_exp_backward if requires_grad else None,
            grad_fn_name="<ExpBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def log(self):

        """
        Element-wise log with base e
        O = log(A)
        dO/dA = 1/a
        """

        output = cp.log(self.data)
   
        def _log_backward(input_grad): 

            if self.requires_grad:
                self_grad = input_grad * (1/self.data)

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad,
                        grad_fn=_log_backward if requires_grad else None, 
                        grad_fn_name="<LogBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            output._add_parents(self)

        return output

    def sum(self, dim=-1, keepdims=False):
        """
        Sum across a dimension.
        Forward: output = self.data.sum(axis=dim, keepdims=keepdims)
        Backward: distribute incoming gradient to all elements along summed axes.
        """
        out_data = self.data.sum(axis=dim, keepdims=keepdims)

        def _sum_backward(input_grad):
            if self.requires_grad:
                # Broadcast input gradient to input shape
                self_grad = cp.broadcast_to(input_grad, self.shape)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_sum_backward if requires_grad else None,
            grad_fn_name="<SumBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def mean(self, dim=-1, keepdims=False):
        """
        Mean across a dimension.
        Forward: output = self.data.mean(axis=dim, keepdims=keepdims)
        Backward: broadcast incoming gradient and divide by number of elements summed.
        """
        out_data = self.data.mean(axis=dim, keepdims=keepdims)

        def _mean_backward(input_grad):

            if self.requires_grad:
                # Compute number of elements reduced over
                dims = dim if isinstance(dim, tuple) else (dim,)
                num_vals_averaged = cupy_prod([self.shape[d] for d in dims])

                # Broadcast upstream gradient and scale
                self_grad = cp.broadcast_to(input_grad, self.shape) / num_vals_averaged
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_mean_backward if requires_grad else None,
            grad_fn_name="<MeanBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def var(self, dim=-1, keepdims=False):
        """
        Variance along a given dimension.
        Var = mean((x - mean(x))^2)
        
        Backward: dVar/dx = 2 * (x - mean(x)) / N * input_grad
        """
        # Forward pass
        mean_vals = self.data.mean(axis=dim, keepdims=True)
        var_vals = ((self.data - mean_vals) ** 2).mean(axis=dim, keepdims=keepdims)

        def _var_backward(input_grad):
            if self.requires_grad:
                # Broadcast input gradient to input shape
                input_grad_broadcast = cp.broadcast_to(input_grad, self.shape)
                
                # Number of elements reduced over
                dims = dim if isinstance(dim, tuple) else (dim,)
                num_vals_reduced = cupy_prod([self.shape[d] for d in dims])
                
                # Gradient formula: 2/N * (x - mean(x)) * upstream gradient
                centered = self.data - mean_vals
                self_grad = 2.0 * centered * input_grad_broadcast / num_vals_reduced

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            var_vals,
            requires_grad=requires_grad,
            grad_fn=_var_backward if requires_grad else None,
            grad_fn_name="<VarBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def max(self, dim=-1, keepdims=False):
        """
        Compute max along axis with autograd support.
        Only propagate gradient to the positions where the maximum occurred.
        """
        out_data = self.data.max(axis=dim, keepdims=keepdims)

        def _max_backward(input_grad):
            
            if self.requires_grad:

                grad = cp.zeros_like(self.data, dtype=cp.float32)

                # Broadcast input_grad if needed
                if dim is not None and not keepdims:
                    input_grad = cp.expand_dims(input_grad, dim)

                # Broadcast to match self shape
                input_grad = input_grad * cp.ones_like(self.data, dtype=cp.float32)
                
                # Only propagate gradient to positions where max occurred
                mask = (self.data == (out_data if keepdims else cp.expand_dims(out_data, dim)))
                grad += input_grad * mask
    
                # Call backward on self
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_max_backward if requires_grad else None,
            grad_fn_name="<MaxBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def argmax(self, dim=-1):
        """
        Compute the indices of the maximum value along a dimension.
        Note: argmax is non-differentiable.
        """
        out_data = self.data.argmax(axis=dim)

        def _argmax_backward(input_grad):
            # No gradient flows through argmax
            return cp.zeros_like(self.data, dtype=cp.float32)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_argmax_backward if requires_grad else None,
            grad_fn_name="<ArgmaxBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def reshape(self, *shape):
        """
        Reshape the tensor. Gradients are reshaped back to the original shape during backprop.
        """
        out_data = self.data.reshape(*shape)

        def _reshape_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_reshape_backward if requires_grad else None,
            grad_fn_name="<ReshapeBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def _toarray(self, input):

        """
        Helper to convert an input to a numpy array
        """
        if isinstance(input, cp.ndarray):
            return input
        elif isinstance(input, Tensor):
            return input.data
        else:           
            return cp.array(input)

    def _add_parents(self, *parents):
        """
        Store references to parent tensors as weakrefs.
        """

        if not isinstance(parents, (list, tuple)):
            parents = (parents)
        self._parents = (weakref.ref(p) for p in parents if p is not None)

    def item(self):
        if self.data.size != 1:
            raise ValueError("only one element tensors can be converted to a Python scalar")
        return self.data.flatten()[0].get().item()

    def astype(self, dtype):
        self.data = self.data.astype(dtype)
        return self
    
    def contiguous(self):
        self.data = cp.ascontiguousarray(self.data, dtype=cp.float32)
        return self
    
    def detach(self):

        detached = Tensor(
            self.data,  
            requires_grad=False,
            grad_fn=None,
            grad_fn_name=None
        )

        return detached