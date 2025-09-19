import math
import cupy as cp
from .tensor import Tensor
# from tensor import Tensor

######################
### Generic Module ###
######################
class Module:

    def parameters(self):
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params
    
    # def parameters(self):
    #     params = []
    #     for val in self.__dict__.values():
    #         if isinstance(val, Tensor):
    #             params.append(val)

    #         if isinstance(val, Linear):
    #             linear_params = val.parameters()
    #             params.extend(linear_params)

    #     return params

    # def __repr__(self):
    #     model_name = self.__class__.__name__
    #     model_string = f"{model_name}(\n"
    #     for key, val in self.__dict__.items():
    #         model_string += f"  ({key}): {val}\n"
    #     model_string += ")"
    #     return model_string

    def _extra_repr(self):
        """
        This extra repr can be overwritten in modules to fill in information
        so layers dont show as Linear() but instead Linear(in_features, out_features)
        """
        return ""

    def _repr(self, indent=0):

        ### We start with 0 indentation for outermost classes ###
        model_name = self.__class__.__name__
        ind = "   " * indent

        ### Grab extra information if we want to include it ###
        extra = self._extra_repr()

        ### If a Module DOESNT CONTAIN any other modules (like our own layers) ###
        if not any(isinstance(v, Module) for v in self.__dict__.values()):
            # leaf module: print in one line
            return f"{ind}{model_name}({extra})\n"

        # Otherwise we have a nested setup and a module contains more modules so #
        # we have to indent it! ####
        s = f"{ind}{model_name}(\n"
        for key, val in self.__dict__.items():
            if isinstance(val, Module):
                s += f"{ind}  ({key}): {val._repr(indent + 1).lstrip()}"
        s += f"{ind})\n"
        return s

    def __repr__(self):
        return self._repr(indent=0).rstrip()

    def __call__(self, *args):
        return self.forward(*args)
    
    def train(self):
        self.training = True
        for val in self.__dict__.values():
            if isinstance(val, Module):
                if hasattr(val, "training"):
                    val.training = self.training

    def eval(self):
        self.training = False
        for val in self.__dict__.values():
            if isinstance(val, Module):
                if hasattr(val, "training"):
                    val.training = self.training


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules = []
        if modules is not None:
            for i, m in enumerate(modules):
                self.append(m)

    def append(self, module):
        assert isinstance(module, Module), "Only Module instances can be added to ModuleList"
        self._modules.append(module)
        # also register it as an attribute so __repr__ and .parameters() work
        setattr(self, str(len(self._modules) - 1), module)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __repr__(self):

        out = "ModuleList([\n"
        for i, layer in enumerate(self._modules):
            out += f"  ({i}): {layer}\n"  # numeric index + layer info
        out += "])"
        return out

#######################
### Random sampling ###
#######################
def rand(shape, low=0.0, high=1.0, dtype=cp.float32):
    return Tensor(cp.random.uniform(low, high, size=shape).astype(dtype))

def randn(shape, mean=0.0, std=1.0, dtype=cp.float32):
    return Tensor(cp.random.normal(mean, std, size=shape).astype(dtype))

def randint(shape, low=0, high=10, dtype=cp.int32):
    return Tensor(cp.random.randint(low, high, size=shape, dtype=dtype))

##############
### LAYERS ###
##############
class AutoLinear(Module):

    def __init__(self, in_features, out_features, bias=True):

        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features

        ### Initialize Weights as Described in nn.Linear ###
        ### https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        k = math.sqrt(1/in_features)
        self.W = Tensor(cp.random.uniform(low=-k, high=k, size=(in_features, out_features), dtype=cp.float32), requires_grad=True)

        if self.bias:
            self.b = Tensor(cp.random.uniform(low=-k, high=k, size=(1, out_features), dtype=cp.float32), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
    
    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"

    def forward(self, x):
        output = x @ self.W

        if self.bias:
            output = output + self.b

        return output

class Linear(Module):
    """
    Linear layer with manual backward, similar to LayerNorm.
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Weight initialization
        k = math.sqrt(1 / in_features)
        self.W = Tensor(
            cp.random.uniform(-k, k, size=(in_features, out_features), dtype=cp.float32),
            requires_grad=True
        )

        if self.bias:
            self.b = Tensor(
                cp.random.uniform(-k, k, size=(1, out_features), dtype=cp.float32),
                requires_grad=True
            )
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: Tensor):
        x_data = cp.ascontiguousarray(x.data)
        W_data = cp.ascontiguousarray(self.W.data)
        out_data = x_data @ W_data

        if self.bias:
            b_data = cp.ascontiguousarray(self.b.data)
            out_data = out_data + b_data

        # Manual backward
        def _linear_backward(grad_output, child):
            grad_output = cp.ascontiguousarray(grad_output)

            # Gradient w.r.t weights
            if self.W.requires_grad:
                grad_W = x_data.T @ grad_output
                self.W.backward(grad_W, child)

            # Gradient w.r.t bias
            if self.bias and self.b.requires_grad:
                grad_b = cp.sum(grad_output, axis=0, keepdims=True)
                self.b.backward(grad_b, child)

            # Gradient w.r.t input x
            if x.requires_grad:
                grad_x = grad_output @ W_data.T
                x.backward(grad_x, child)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.W.requires_grad or (self.bias and self.b.requires_grad),
            grad_fn=_linear_backward,
            grad_fn_name="<LinearBackward>"
        )

        # Add children for autograd graph
        x._add_child(out)
        self.W._add_child(out)
        if self.bias:
            self.b._add_child(out)

        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"

    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Tensor((cp.random.randn(num_embeddings, embedding_dim) / cp.sqrt(num_embeddings)).astype(cp.float32), requires_grad=True)

    def __call__(self, indices):
        return self.forward(indices)
    
    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
    
    def _extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
    
    def forward(self, indices):
        return self.weight[indices]

class Dropout(Module):
    def __init__(self, dropout_p=0.5):
        self.p = dropout_p
        self.training = True
        self.mask = None  # will hold mask during forward

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dropout(p={self.p})"

    def forward(self, x):
        if not self.training:
            return x  # no dropout during evaluation

        # Generate dropout mask (random 0/1), then scale
        self.mask = (cp.random.rand(*x.shape) >= self.p).astype(cp.float32)
        self.mask = self.mask / (1.0 - self.p)  # scale remaining activations

        # Apply mask
        out = x * self.mask
        return out

class AutoLayerNorm(Module):
    def __init__(self, embed_dim):

        self.embed_dim = embed_dim
        self.gamma = Tensor(cp.ones(shape=(1,1,embed_dim), dtype=cp.float32), requires_grad=True)
        self.beta = Tensor(cp.zeros(shape=(1,1,embed_dim), dtype=cp.float32), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"LayerNorm({self.embed_dim})"
    
    def _extra_repr(self):
        return f"{self.embed_dim}"
    
    def forward(self, x):
        var_x = x.var(dim=-1, keepdims=True)
        norm_x = (x - x.mean(dim=-1, keepdims=True)) / var_x**0.5
        return norm_x * self.gamma + self.beta

class LayerNorm(Module):
    """
    Optimized LayerNorm with manual backward for contiguous memory.
    """
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = Tensor(cp.ones((1, 1, embed_dim), dtype=cp.float32), requires_grad=True)
        self.beta = Tensor(cp.zeros((1, 1, embed_dim), dtype=cp.float32), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_data = cp.ascontiguousarray(x.data)

        # Compute mean and variance along last dim
        mean = cp.mean(x_data, axis=-1, keepdims=True)
        var = cp.var(x_data, axis=-1, keepdims=True)
        inv_std = cp.reciprocal(cp.sqrt(var + self.eps))

        # Normalize
        norm_x = (x_data - mean) * inv_std

        # Scale and shift
        gamma_data = cp.ascontiguousarray(self.gamma.data)
        beta_data = cp.ascontiguousarray(self.beta.data)
        out_data = norm_x * gamma_data + beta_data

        # Define manual backward
        def _layernorm_backward(grad_output, child):
            grad_output = cp.ascontiguousarray(grad_output)
            N = x_data.shape[-1]

            # Grad w.r.t gamma and beta
            if self.gamma.requires_grad:
                grad_gamma = cp.sum(grad_output * norm_x, axis=(0,1), keepdims=True)
                self.gamma.backward(grad_gamma, child)
            if self.beta.requires_grad:
                grad_beta = cp.sum(grad_output, axis=(0,1), keepdims=True)
                self.beta.backward(grad_beta, child)

            # Grad w.r.t input x
            if x.requires_grad:
                grad_norm = grad_output * gamma_data
                mean_grad = cp.mean(grad_norm, axis=-1, keepdims=True)
                mean_norm_grad = cp.mean(grad_norm * norm_x, axis=-1, keepdims=True)

                grad_input = (grad_norm - mean_grad - norm_x * mean_norm_grad) * inv_std
                x.backward(grad_input, child)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
            grad_fn=_layernorm_backward,
            grad_fn_name="<LayerNormBackward>"
        )

        # Add children
        x._add_child(out)
        self.gamma._add_child(out)
        self.beta._add_child(out)

        return out

    def __repr__(self):
        return f"LayerNorm({self.embed_dim})"

    def _extra_repr(self):
        return f"{self.embed_dim}"
    

############################
### Activation Functions ###
############################
class Sigmoid:
    def __init__(self):
        pass

    def __repr__(self):
        return "Sigmoid()"
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return 1 / (1 + (-x).exp())

class AutoReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return "ReLU()"
    
    def forward(self, x):

        ### Check where x < 0 ###
        mask = Tensor(cp.where(x.data < 0, 0, 1))
        x = x * mask
        return x
    
class ReLU(Module):
    """
    Optimized ReLU with manual backward and contiguous memory.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_data = cp.ascontiguousarray(x.data)
        mask = x_data > 0  # boolean mask
        mask = cp.ascontiguousarray(mask.astype(cp.float32))

        out_data = x_data * mask

        # Manual backward
        def _relu_backward(grad_output, child):
            grad_output = cp.ascontiguousarray(grad_output)
            grad_input = grad_output * mask
            x.backward(grad_input, child)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            grad_fn=_relu_backward,
            grad_fn_name="<ReLUBackward>"
        )

        # Add child for autograd
        x._add_child(out)

        return out

    def __repr__(self):
        return "ReLU()"

class AutoSoftmax:
    def __init__(self):
        pass

    def __call__(self, x, dim=-1):
        return self.forward(x, dim=dim)
    
    def __repr__(self):
        return "Softmax()"
    
    def forward(self, x, dim=-1):
        x = x.exp()
        x = x / x.sum(dim=dim, keepdims=True)
        return x
    
class Softmax(Module):
    """
    Optimized Softmax with manual backward and contiguous memory.
    """
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return self.forward(x, dim)

    def forward(self, x, dim):

        self.dim = dim
        x_data = cp.ascontiguousarray(x.data)

        # Numerical stability: subtract max along dim
        max_val = cp.max(x_data, axis=self.dim, keepdims=True)
        shifted = x_data - max_val
        exp_x = cp.exp(shifted)
        sum_exp = cp.sum(exp_x, axis=self.dim, keepdims=True)
        out_data = exp_x / sum_exp

        # Define manual backward
        def _softmax_backward(grad_output, child):
            grad_output = cp.ascontiguousarray(grad_output)

            if x.requires_grad:
                # Softmax derivative: grad_input = s * (grad - sum(grad*s))
                s = out_data
                sum_grad_s = cp.sum(grad_output * s, axis=self.dim, keepdims=True)
                grad_input = s * (grad_output - sum_grad_s)
                x.backward(grad_input, child)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            grad_fn=_softmax_backward,
            grad_fn_name="<SoftmaxBackward>"
        )

        # Add child to autograd graph
        x._add_child(out)

        return out

    def __repr__(self):
        return f"Softmax(dim={self.dim})"

###################### 
### Loss Functions ###
######################

class AutoCrossEntropyLoss(Module):
    """ Cross Entropy Loss class, returns the loss given the output and the expected indexes. """
    def __init__(self):
        super().__init__()

    def __call__(self, z, y):
        return self.forward(z, y)

    def forward(self, z, y):

        ### Reshape Logits to (-1 x C) ###
        *other_dims, num_classes = z.shape
        
        ### Get Total Flattened Dimension ###
        B = 1
        for dim in other_dims:
            B*= dim

        ### Reshape Data to (-1, Num Classes)
        z = z.reshape(B, num_classes)
        
        ### Flatten Y to B as well if needed ###
        if len(y.shape) != 1:
            y = y.reshape(B)

        ### Stable Log-Softmax ###
        # Step 1: subtract max for numerical stability
        z_shifted = z - z.max(dim=1, keepdims=True)

        # Step 2: compute log-sum-exp
        logsumexp = (z_shifted.exp()).sum(dim=1, keepdims=True).log()

        # Step 3: log-softmax
        log_softmax = z_shifted - logsumexp

        ### Negative log-likelihood for correct class ###
        nll = -log_softmax[cp.arange(B), y]

        ### Mean loss ###
        loss = nll.sum() / B

        return loss

class CrossEntropyLoss(Module):
    """
    Cross-entropy loss with manual backward, numerical stability, and contiguous memory.
    Expects logits (z) and integer class labels (y).
    """
    def __init__(self):
        super().__init__()

    def __call__(self, z, y):
        return self.forward(z, y)

    def forward(self, z, y):
        # Flatten z and y if necessary
        *other_dims, num_classes = z.shape
        B = 1
        for dim in other_dims:
            B *= dim

        z_data = cp.ascontiguousarray(z.data.reshape(B, num_classes))
        y_data = cp.ascontiguousarray(y.data.reshape(B)) if isinstance(y, Tensor) else cp.ascontiguousarray(y.reshape(B))

        # Stable log-softmax
        z_shifted = z_data - cp.max(z_data, axis=1, keepdims=True)
        logsumexp = cp.log(cp.sum(cp.exp(z_shifted), axis=1, keepdims=True))
        log_softmax = z_shifted - logsumexp

        # Negative log-likelihood
        nll = -log_softmax[cp.arange(B), y_data]
        loss_value = cp.sum(nll) / B

        # Define manual backward
        def _cross_entropy_backward(grad_output, child):
            grad_output = float(grad_output)  # scalar from loss

            if z.requires_grad:
                # Softmax probabilities
                softmax = cp.exp(log_softmax)  # shape (B, C)
                grad_input = softmax
                grad_input[cp.arange(B), y_data] -= 1
                grad_input *= grad_output / B  # scale by grad_output / batch_size

                z.backward(grad_input.reshape(*z.shape), child)

        out = Tensor(
            cp.array(loss_value, dtype=cp.float32),
            requires_grad=z.requires_grad,
            grad_fn=_cross_entropy_backward,
            grad_fn_name="<CrossEntropyBackward>"
        )

        # Add child for autograd
        z._add_child(out)

        return out

class MSELoss:

    def __init__(self):
        pass

    def __call__(self, pred, labels):
        return self.forward(pred, labels)
    
    def forward(self, pred, labels):
        return ((pred-labels)**2).mean(dim=0)



