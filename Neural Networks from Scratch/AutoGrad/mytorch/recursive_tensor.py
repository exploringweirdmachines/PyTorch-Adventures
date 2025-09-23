"""
Recursive Backward implementation of Autograd
"""

import os
import cupy as cp
import weakref
from contextlib import contextmanager

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
        self.children = []

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

        for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)):
            ### If our tensor dim is 1 but the grad dim is not, accumulate on that dimension ###
            if (x_dim == 1) and (grad_dim != 1):
                x_grad = x_grad.sum(axis=idx, keepdims=True)
            ### Otherwise verify that our x_dim and grad dim are the same!! 
            else:
                assert (x_dim == grad_dim)

        return x_grad
    
    def backward(self, input_grad=None, child=None):
        
        ### During inference the output has no grads, you cant call backward on this ###
        if child is None and self.grad_fn is None:
            raise RuntimeError(
                "No grad_fn defined for this tensor."
            )
        
        if self.requires_grad:

            ### Base Case (dL/dL = 1) for the start of chain rule ###
            if input_grad is None:
                input_grad = cp.ones_like(self.data, dtype=cp.float32)

            ### Accumulate Gradients ###
            if not input_grad.flags.c_contiguous:
                input_grad = cp.ascontiguousarray(input_grad)

            ### If self.grad is None, allocate memory ###
            if self.grad is None:
                self.grad = cp.zeros_like(self.data, dtype=cp.float32)
            self.grad += input_grad

            ### We are exhausting this backprop path from "child", so we can pop child out ###
            if child is not None:
                self.children = [wr for wr in self.children if wr() is not child]
                # self.children = [c for c in self.children if c is not child]
                # self.children.remove(child) # this breaks __eq__

            #### NOTE: THIS METHOD THROWS OOM ###   
            # ### If we have a grad function, we can do backward pass ###
            # if self.grad_fn is not None:
            #     ### Until we exhast all the children we cannot move backwards ###
            #     ### If a single tensor has multiple children, we must backward all the children ###
            #     ### and accumulate gradients for tensor before backwarding again ###
            #     if len(self.children) == 0:
            #         self.grad_fn(self.grad, self)

            ### PROBLEM: We have multiple references happening here. 
            ### 1) Child->Parent reference: when we do c = a + b, the resulting tensor c stores 
            ###    a grad function _add_backward(), so this function hold a reference to its parent tensors
            ###    so it can use that data in the backward pass
            ### 2) Parent->Child reference: Simulteously, the parent tensors a and b store a reference to their
            ###    child tensor c in their self.children list. 

            ### This creates a weird cycle of references a -> c -> _add_backward -> a
            ### Because of this, python garbage collector doesnt really know if its safe to delete even 
            ### after the backward pass is done. And so with every iteration, a new graph is created and leaked

            ### Solution: Break this cycle. Explcitly break it by clearing the grad function once during the backward pass
            ###           A node only needs the grad_fn one time, afterwards it can be removed

            if self.grad_fn is not None and not self.children:
                ### Store the function to be called in a temporary variable ### 
                ### because this variable is not tied to the Tensor object it will ###
                ### immediately be deleted ###
                grad_fn_to_call = self.grad_fn
                
                ### Clear Grad Function so the Garbage Collector can delete it ###
                self.grad_fn = None

                # Call the backward function
                grad_fn_to_call(self.grad, self)

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
        def _add_backward(input_grad, child):

            if self.requires_grad:
                self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                self.backward(self_grad, child)
            if val.requires_grad:
                val_grad = input_grad
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                val.backward(val_grad, child)

        ### Wrap our output in our tensor object ###
        ### We will compute grad on this if self or val are also tracking gradients ###
        ### We also store the backward_fn for the backward pass ###
        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_add_backward if requires_grad else None,
                        grad_fn_name="<AddBackward>" if requires_grad else None)#.astype(cp.float32)
        
        ### This output is the child of the inputs a and b ###
        if requires_grad:
            self._add_child(output)
            val._add_child(output)
        
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
        def _sub_backward(input_grad, child):
            if self.requires_grad:
                self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                self.backward(self_grad, child)
                
            if val.requires_grad:
                val_grad = -input_grad
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                val.backward(val_grad, child)

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
            self._add_child(output)
            val._add_child(output)
        
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

        def _mul_backward(input_grad, child):

            if self.requires_grad:
                self_grad = input_grad * val.data
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                self.backward(self_grad, child)
            
            if val.requires_grad:
                val_grad = input_grad * self.data
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                val.backward(val_grad, child)

        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_mul_backward if requires_grad else None,
                        grad_fn_name="<MulBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)
            val._add_child(output)

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
        output = cp.matmul(self.data, val.data, out=prealloc)

        def _matmul_backward(input_grad, child):

            if self.requires_grad:
                prealloc_grad_self = cp.empty(shape=self.shape, dtype=self.data.dtype)
                grad_self = cp.matmul(input_grad, val.data.swapaxes(-1, -2), out=prealloc_grad_self)
                self.backward(grad_self, child)

            if val.requires_grad:
                prealloc_grad_val = cp.empty(shape=val.shape, dtype=self.data.dtype)
                grad_val = cp.matmul(self.data.swapaxes(-1, -2), input_grad, out=prealloc_grad_val)
                val.backward(grad_val, child)

        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(
            output,
            requires_grad=requires_grad,
            grad_fn=_matmul_backward if requires_grad else None,
            grad_fn_name="<MatmulBackward>" if requires_grad else None
        )

        if requires_grad:
            self._add_child(output)
            val._add_child(output)

        return output
        
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
  
        def _div_backward(input_grad, child):
            if self.requires_grad:
                self_grad = input_grad / val.data
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                self.backward(self_grad, child)

            if val.requires_grad:
                val_grad = input_grad * -1 * self.data / (val.data**2)
                val_grad = self._broadcasted_grad_accumulate(val.shape, val_grad)
                val.backward(val_grad, child)
        
        ### Convert to Tensor ###
        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_div_backward if requires_grad else None,
                        grad_fn_name="<DivBackward>" if requires_grad else None)#.astype(cp.float32) 

        ### This output is the child of the inputs a and b ###
        if requires_grad:
            self._add_child(output)
            val._add_child(output)

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
    
        def _pow_backward(input_grad, child):
            self_grad = input_grad * (exponent * self.data ** (exponent-1))
            self.backward(self_grad, child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_pow_backward if requires_grad else None,
                        grad_fn_name="<PowBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output

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
        output = self.data.swapaxes(dim1, dim2)
 
        def _transpose_backward(input_grad, child):
            # Just swap back the same two dims
            if self.requires_grad:
                self.backward(input_grad.swapaxes(dim1, dim2), child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_transpose_backward if requires_grad else None,
                        grad_fn_name="<TransposeBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output
    
    def permute(self, *dims):
        """
        Permute tensor dimensions according to dims.
        Example: (0, 2, 1) will reorder axes in that order.
        """
        output = cp.transpose(self.data, axes=dims)

        def _permute_backward(input_grad, child):

            if self.requires_grad:
                # Inverse permutation: figure out where each axis went
                inv_dims = cp.argsort(dims)
                self.backward(cp.transpose(input_grad, axes=inv_dims), child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                    requires_grad=requires_grad,
                    grad_fn=_permute_backward if requires_grad else None,
                    grad_fn_name="<PermuteBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)
        return output

    def exp(self):

        """
        Element-wise exponentiation of the base e
        O = e^A
        dO/dA = e^A
        """

        output = cp.exp(self.data)

        def _exp_backward(input_grad, child):  

            if self.requires_grad: 
                self_grad = input_grad * cp.exp(self.data)
                self.backward(self_grad, child)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad,
                        grad_fn=_exp_backward if requires_grad else None, 
                        grad_fn_name="<ExpBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output
    
    def log(self):

        """
        Element-wise log with base e
        O = log(A)
        dO/dA = 1/a
        """

        output = cp.log(self.data)
   
        def _log_backward(input_grad, child):  

            if self.requires_grad: 
                self_grad = input_grad * (1/self.data)
                self.backward(self_grad, child)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad,
                        grad_fn=_log_backward if requires_grad else None, 
                        grad_fn_name="<LogBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output

    def sum(self, dim=-1, keepdims=False):

        """
        Sum across a dimension!

        O = sum([a_1, a_2, a_3, ...])

        Remember, sum operations just channel the incoming gradients from the later computational paths. 
        This means our input_gradient coming from operations after the sum here just needs to be copied to all 
        values of [a_1, a_2, a_3, ...]

        dO/da_1 = input_grad
        dO/da_2 = input_grad
        dO/da_3 = input_grad
        ...

        """

        output = self.data.sum(axis=dim, keepdims=keepdims)

        def _sum_backward(input_grad, child):

            if self.requires_grad:
                # ### Add dimensions to input grad to match self
                # grad_dims = len(input_grad.shape)
                # self_dims = len(self.shape)
                
                # ### Expand input grad (upstream) to match dimensions of the input ###
                # if grad_dims != self_dims:
                #     diff = self_dims - grad_dims    
                #     for _ in range(diff):
                #         input_grad = cp.expand_dims(input_grad, axis=-1)

                # ### Copies gradients from input to all values in the input ###
                # self_grad = input_grad * cp.ones((self.shape))
                
                ### Simpler to just use broadcast_to, to match shapes
                self_grad = cp.broadcast_to(input_grad, self.shape)
                self.backward(self_grad, child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sum_backward if requires_grad else None,
                        grad_fn_name="<SumBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output
    
    def mean(self, dim=-1, keepdims=False):

        """
        Almost identical to Sum across a dimension, except divided by the constant of the number of elements summed

        O = sum([a_1, a_2, a_3, ..., a_N]) / N

        Remember, sum operations just channel the incoming gradients from the later computational paths. 
        This means our input_gradient coming from operations after the sum here just needs to be copied to all 
        values of [a_1, a_2, a_3, ...]

        dO/da_1 = input_grad/N
        dO/da_2 = input_grad/N
        dO/da_3 = input_grad/N
        ...

        """
        
        output = self.data.mean(axis=dim, keepdims=keepdims)

        def _mean_backward(input_grad, child, dim=dim):
                
                if self.requires_grad:
                    # ### Add dimensions to input grad to match self
                    # grad_dims = len(input_grad.shape)
                    # self_dims = len(self.shape)

                    # ### Expand input grad (upstream) to match dimensions of the input ###
                    # if grad_dims != self_dims:
                    #     diff = self_dims - grad_dims    
                    #     for _ in range(diff):
                    #         input_grad = cp.expand_dims(input_grad, axis=-1)

                    # ### We average over the dim dimension ###
                    # ### and averaging is just a sum / constant ###
                    # ### where the constant is the num elements in that dim ###
                    # ### So we can multiply our input_grad by the 1/constant ###
                    # if isinstance(dim, int):
                    #     dim = [dim]
                    # elif isinstance(dim, tuple):
                    #     dim = list(dim)
                    
                    # ### Multiply together all flattened dimensions sizes to get num_vals_averaged ###
                    # dim_sizes = [self.shape[i] for i in dim]
                    # num_vals_averaged = 1
                    # for dim in dim_sizes:
                    #     num_vals_averaged*= dim

                    # ### Copy 1/num_vals_averaged to all values in the inputs ###
                    # self_grad = input_grad * cp.ones((self.shape)) / num_vals_averaged
    
                    ### Easier to just use broadcast_to ###
                    self_grad = cp.broadcast_to(input_grad, self.shape) / cupy_prod([self.shape[d] for d in (dim if isinstance(dim, tuple) else [dim])])
                    self.backward(self_grad, child)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_mean_backward if requires_grad else None,
                        grad_fn_name="<MeanBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output
    
    def var(self, dim=-1, keepdims=False):
        """
        Variance along a given dimension.

        Var = mean((x - mean(x))^2)

        Backward:
        dVar/dx = 2 * (x - mean(x)) / N * input_grad
        """
        
        # Compute forward pass using cupy
        mean_vals = self.data.mean(axis=dim, keepdims=True)
        var_vals = ((self.data - mean_vals) ** 2).mean(axis=dim, keepdims=keepdims)

        def _var_backward(input_grad, child, dim=dim, mean_vals=mean_vals):
            
            if self.requires_grad:
            
                # ### Add dimensions to input grad to match self
                # grad_dims = len(input_grad.shape)
                # self_dims = len(self.shape)

                # ### Expand input grad (upstream) to match dimensions of the input ###
                # if grad_dims != self_dims:
                #     diff = self_dims - grad_dims    
                #     for _ in range(diff):
                #         input_grad = cp.expand_dims(input_grad, axis=-1)

                # ### We compute variance over the dim dimension ###
                # ### and so we need the number of elements this happened on! ###
                # if isinstance(dim, int):
                #     dim = [dim]
                # elif isinstance(dim, tuple):
                #     dim = list(dim)

                # ### Multiply together all flattened dimensions sizes ###
                # dim_sizes = [self.shape[i] for i in dim]
                # num_vals_vared = 1
                # for dim in dim_sizes:
                #     num_vals_vared*= dim

                # # Gradient formula: dVar/dx = 2/N * (x - mean(x)) * input_grad
                # centered = self.data - mean_vals
                # self_grad = (2.0 / num_vals_vared) * centered * input_grad

                ### Easier to just use broadcast_to ###
                centered = self.data - mean_vals
                self_grad = 2.0 * centered * cp.broadcast_to(input_grad, self.shape) / cupy_prod([self.shape[d] for d in (dim if isinstance(dim, tuple) else [dim])])
                self.backward(self_grad, child)
    
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(var_vals,
                        requires_grad=requires_grad,
                        grad_fn=_var_backward if requires_grad else None,
                        grad_fn_name="<VarBackward>" if requires_grad else None)#.astype(cp.float32)

        if requires_grad:
            self._add_child(output)

        return output

    def max(self, dim=-1, keepdims=False):
        """
        Compute max along axis (like numpy.max) with autograd support.
        """
        output = self.data.max(axis=dim, keepdims=keepdims)

        def _max_backward(input_grad, child):
            
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
                self.backward(grad, child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                    requires_grad=requires_grad,
                    grad_fn=_max_backward if requires_grad else None,
                    grad_fn_name="<MaxBackward>" if requires_grad else None)#.astype(cp.float32)

        if requires_grad:
            self._add_child(output)

        return output
    
    def argmax(self, dim=-1):
        
        output = self.data.argmax(axis=dim)

        def _argmax_backward(input_grad, child):
            """
            argmax has no derivative!
            """
            return cp.zeros_like(self)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_argmax_backward if requires_grad else None, 
                        grad_fn_name="<ArgmaxBackward>" if requires_grad else None)
        
        return output
    
    def reshape(self, *shape):

        """
        If we reshape our tensor, we just need to reshape the incoming identically! 
        Remember, gradients of a tensor are the same shape as the tensor itself, so we 
        just need to make sure that our gradient index coorespond to the correct tensor index. 
        """
        
        output = self.data.reshape(*shape)
   
        def _reshape_backward(input_grad, child):

            if self.requires_grad:
                self_grad = input_grad.reshape(self.data.shape)
                self.backward(self_grad, child)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_reshape_backward if requires_grad else None, 
                        grad_fn_name="<ReshapeBackward>" if requires_grad else None)
        
        if requires_grad:
            self._add_child(output)
        
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
        output = self.data[idx]

        def _index_backward(input_grad, child):
            
            if self.requires_grad:
                grad = cp.zeros_like(self.data, dtype=cp.float32)
                grad[idx] += input_grad
                self.backward(grad, child)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_index_backward if requires_grad else None,
                        grad_fn_name="<IndexBackward>" if requires_grad else None)#.astype(cp.float32)
        
        if requires_grad:
            self._add_child(output)

        return output

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

    # def _add_child(self, child_tensor):

    #     """
    #     Helper function to add a tensor as a child of an operation
    #     """
        
    #     if not isinstance(child_tensor, Tensor):
    #         raise Exception("Children of Tensors must also be a Tensor")
    #     self.children.append(child_tensor)

    def _add_child(self, *children):
        """
        Helper function to add a tensor as a child of an operation using Weakreferences
        You can learn more about this here: https://martinheinz.dev/blog/112
        """

        if not isinstance(children, (list, tuple)):
            children = [children]

        self.children.extend([weakref.ref(p) for p in children if p is not None])
    
    def item(self):
        if self.data.size != 1:
            raise ValueError("only one element tensors can be converted to a Python scalar")
        return self.data.flatten()[0].get().item()

    def astype(self, dtype):
        self.data = self.data.astype(dtype)
        return self
