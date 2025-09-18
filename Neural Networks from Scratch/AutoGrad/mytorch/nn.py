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
class Linear(Module):

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
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    def _extra_repr(self):
        return f"p={self.p}"

    def forward(self, x):
        if not self.training:
            return x
        
        ### Generate Mask 3##
        noise = rand(x.shape)
        mask = (noise > self.p)
        mask.astype(cp.float32)

        ### Multiply Data by Mask ###
        x = x * mask
        x = x / (1 - self.p)

        return x

class LayerNorm(Module):
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

class ReLU:
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

class Softmax:
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

###################### 
### Loss Functions ###
######################

class CrossEntropyLoss(Module):
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

class MSELoss:

    def __init__(self):
        pass

    def __call__(self, pred, labels):
        return self.forward(pred, labels)
    
    def forward(self, pred, labels):
        return ((pred-labels)**2).mean(dim=0)
