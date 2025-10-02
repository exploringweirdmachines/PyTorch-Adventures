from .tensor import Tensor

### Functional Access to Non-Dunder Methods ###
def transpose(input, dim1, dim2):
    return input.transpose(dim1, dim2)

def permute(input, *dims):
    return input.permute(dims)

def reshape(input, *shape):
    return input.reshape(shape)

def exp(input):
    return input.exp()

def log(input):
    return input.log()

def sum(input, dim=None, keepdims=False):
    return input.sum(dim, keepdims)

def mean(input, dim=None, keepdims=False):
    return input.mean(dim, keepdims)

def var(input, dim=None, keepdims=False):
    return input.var(dim, keepdims)

def max(input, dim=None, keepdims=False):
    return input.max(dim, keepdims)

def argmax(input, dim=None, keepdims=False):
    return input.argmax(dim)

### Additional Ops ###
def chunk(input, chunks, dim=0):
    """
    Split a tensor into `chunks` along dimension `dim`.
    Returns a list of Tensors.

    to make this easy we use slices

    a = ["a", "b", "c", "d", "e", "f", "g"]
    a[1:3] = ["b", "c", "d"]
    a[slice(1,3)] = ["b", "c", "d"]
    """
    size = input.shape[dim]
    if size % chunks != 0:
        raise ValueError(f"Cannot split dimension {dim} of size {size} into {chunks} equal chunks")
    
    chunk_size = size // chunks
    out_tensors = []

    for i in range(chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size

        # Slice the underlying array directly
        idx = [slice(None)] * input.ndim
        idx[dim] = slice(start, end)
        slice_data = input.data[tuple(idx)]

        def _chunk_backward(input_grad, start=start, end=end):
            if input.requires_grad:
                grad = input.xp.zeros_like(input.data, dtype=input.data.dtype)
                
                # Ensure input_grad has the correct shape
                grad_slice_shape = list(grad.shape)
                grad_slice_shape[dim] = end - start
                grad_slice = input_grad.reshape(grad_slice_shape)

                # Insert gradient slice into the right position
                grad_idx = [slice(None)] * grad.ndim
                grad_idx[dim] = slice(start, end)
                grad[tuple(grad_idx)] = grad_slice

                # Accumulate gradients
                if input.grad is None:
                    input.grad = grad
                else:
                    input.grad += grad

        requires_grad = input.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            slice_data,
            requires_grad=requires_grad,
            grad_fn=_chunk_backward if requires_grad else None,
            grad_fn_name="<ChunkBackward>" if requires_grad else None,
            device=input.device
        )

        if requires_grad:
            out._add_parents(input)

        out_tensors.append(out)

    return out_tensors

def concatenate():
    pass

def stack():
    pass

def masked_fill():
    pass