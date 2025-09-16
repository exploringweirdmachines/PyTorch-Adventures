import cupy as np
import math

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
class GradTensor:
    def __init__(self, params):
        self.params = params
        self.shape = params.shape
        self.grad = None
    
    def _zero_grad(self):
        self.grad = None

### STANDARD LAYERS ###
class Linear(Layer):
    """
    Basic Implementation of the Linear Layer following nn.Linear
    y = xW^T + b
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        ### Initialization to Match nn.Linear ###
        k = 1 / self.in_features

        self.weight = GradTensor(
            np.random.uniform(
                low=-math.sqrt(k),
                high=math.sqrt(k), 
                size=(in_features, out_features)
            )
        )

        self.bias = None
        if bias:
            self.bias = GradTensor(
                np.random.uniform(
                    low=-math.sqrt(k), 
                    high=math.sqrt(k), 
                    size=(1, out_features)
                )
            )

    def forward(self, x):
        # For backprop, dL/dW will need X^t, so save X for future use 
        self.x = x

        # X has shape (B x in_features), w has shape (in_feature x out_features)
        x = x @ self.weight.params

        if self.bias is not None:
            x = x + self.bias.params

        return x
    
    def backward(self, output_grad):
        ### Derivative w.r.t. W: X^t @ output_grad 
        self.weight.grad = self.x.T @ output_grad

        ### Derivative of Bias is just the output grad summed along batch 
        if self.bias is not None:
            self.bias.grad = output_grad.sum(axis=0, keepdims=True)

        ### We need derivative w.r.t input for the next step 
        input_grad = output_grad @ self.weight.params.T

        return input_grad

class MSELoss:
    """
    L = E[(y-y_hat)^2]
    """
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred)**2)
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        grad = -(2/batch_size) * (self.y_true - self.y_pred)
        return grad

class SLOW_SoftMax:
    """
    Softmax normalization to convert logits -> probabilities
    Backward explicitly computes the jacobian for each sample
    """
    def forward(self, x):
        self.x = x
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, output_grad):
        gradient = np.zeros_like(self.probs)
        batch_size = len(self.x)
        for i in range(batch_size):
            sample_probs = self.probs[i]
            j = -sample_probs.reshape(-1,1) * sample_probs.reshape(1,-1)
            j[np.diag_indices(j.shape[0])] = sample_probs * (1 - sample_probs)
            a = output_grad[i] @ j
            gradient[i] = a
       
        return gradient

class SoftMax:
    """
    Softmax normalization to convert logits -> probabilities
    Backward vectorizes gradient computation
    """
    def forward(self, x):
        self.x = x
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, output_grad):
        dot_product = np.sum(output_grad * self.probs, axis=-1, keepdims=True)
        gradient = self.probs * (output_grad - dot_product)
        return gradient
    
class SLOW_CrossEntropyLoss:
    """
    Cross Entropy Loss that assumes inputs are probabilities (already softmaxed).
    Targets are integer class indices (no one-hot).
    """
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) integer class indices
        y_pred: (batch_size, num_classes) probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        batch_size = y_pred.shape[0]
        # Pick probability of the correct class for each sample
        correct_class_probs = y_pred[np.arange(batch_size), y_true]
        # Compute loss
        loss = -np.mean(np.log(correct_class_probs + 1e-12))  # epsilon for safety
        return loss

    def backward(self):
        """
        Gradient wrt probabilities
        dL/dy_pred = -y_true / y_pred
        But since y_true is an index, we only subtract at the correct class.
        """
        batch_size, num_classes = self.y_pred.shape
        grad = np.zeros_like(self.y_pred)
        # Only the correct class contributes to loss: -y_i / p_c
        grad[np.arange(batch_size), self.y_true] = -1 / (self.y_pred[np.arange(batch_size), self.y_true] + 1e-12)
        return grad

class CrossEntropyLoss:
    def forward(self, y_true, logits):
        """
        y_true: [batch_size] integer class indices
        logits: [batch_size, num_classes] raw scores (NOT softmaxed)
        """
        self.y_true = y_true
        self.logits = logits
        # numerically stable softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # pick the probabilities of the correct class for each sample
        batch_size = y_true.shape[0]
        correct_class_probs = self.probs[np.arange(batch_size), y_true]
        # cross-entropy loss
        loss = -np.mean(np.log(correct_class_probs + 1e-12))
        return loss

    def backward(self):
        """
        Gradient of loss wrt logits (softmax + cross entropy simplified)
        """
        batch_size, num_classes = self.logits.shape
        grad = self.probs.copy()
        # subtract 1 at the correct class index
        grad[np.arange(batch_size), self.y_true] -= 1
        return grad

class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, output_grad):
        sigmoid_x = self.forward(self.x)
        sigmoid_grad = sigmoid_x * (1 - sigmoid_x)
        input_grad = sigmoid_grad * output_grad
        return input_grad

class ReLU:
    def forward(self, x):
        self.x = x
        return np.clip(x, a_min=0, a_max=None)
    
    def backward(self, output_grad):
        grad = np.zeros_like(self.x)
        grad[self.x > 0] = 1
        return grad * output_grad

### SOME CONVOLUTION OPS ###
class SLOW_Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        ### Xavier Initialization
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        self.weight = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        batch_size, c, h, w = x.shape

        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            self.x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), "constant")
        else:
            self.x_padded = x

        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride + self.kernel_size
                        w_start, w_end = j*self.stride, j*self.stride + self.kernel_size
                        
                        ### Computed 1 Output Channel At a Time ###
                        out[b, oc, i, j] = np.sum(
                            self.x_padded[b, :, h_start:h_end, w_start:w_end] * self.weight[oc]
                        ) + self.bias[oc]

        return out
    
    def backward(self, grad_out):
        batch_size, _, h_padded, w_padded = self.x_padded.shape
        grad_x_padded = np.zeros_like(self.x_padded)
        grad_w = np.zeros_like(self.weight)
        grad_b = np.zeros_like(self.bias)

        out_h, out_w = grad_out.shape[2], grad_out.shape[3]

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride + self.kernel_size
                        w_start, w_end = j*self.stride, j*self.stride + self.kernel_size

                        grad_w[oc] += grad_out[b, oc, i, j] * self.x_padded[b, :, h_start:h_end, w_start:w_end]     
                        grad_x_padded[b, :, h_start:h_end, w_start:w_end] += grad_out[b, oc, i, j] * self.weight[oc]
                
                grad_b[oc] += np.sum(grad_out[b,oc])
        
        if self.padding > 0:
            grad_x = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_x = grad_x_padded

        self.grad_w = grad_w
        self.grad_b = grad_b

        return grad_x
    
class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier Initialization
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        weights = np.random.uniform(-limit, limit, 
                                    (out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            bias = np.zeros((1, out_channels)) 

        # Create Linear Layer #
        self.conv_linear = Linear(in_channels * kernel_size * kernel_size, out_channels, bias=bias is not None)

        # Set Weights of Linear Layer with our Xavier Init ###
        self.conv_linear.weight.params = weights.reshape(in_channels * kernel_size * kernel_size, out_channels)
        if bias:
            self.conv_linear.bias.params = bias

    def im2col(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if P > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        else:
            x_padded = x

        out_h = (H + 2*P - K)//S + 1
        out_w = (W + 2*P - K)//S + 1
        n_patches = out_h * out_w

        ### Extract image patches for mat mul (convolution) ###
        cols = np.zeros((B, n_patches, C*K*K))
        for i in range(out_h):
            for j in range(out_w):
                patch = x_padded[:, :, i*S:i*S+K, j*S:j*S+K]
                cols[:, i*out_w + j, :] = patch.reshape(B, -1)

        return cols, out_h, out_w

    def col2im(self, cols, x_shape, out_h, out_w):
        B, C, H, W = x_shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if P > 0:
            x_padded = np.zeros((B, C, H + 2*P, W + 2*P))
        else:
            x_padded = np.zeros((B, C, H, W))

        for i in range(out_h):
            for j in range(out_w):
                ### Extract corresponding column that goes with this patch in the ####
                ### original image. This will be used in our backward pass on our gradients ###
                ### to return our grad tensor from a simple matrix (for matmul in linear) ###
                ### to the normal shape of a convolution. Due to the overlap between patches 
                ### we want to accumulate up all the overlapping contributions to gradients ###
                ### Hence we use += ###
                patch = cols[:, i*out_w + j, :].reshape(B, C, K, K)
                x_padded[:, :, i*S:i*S+K, j*S:j*S+K] += patch

        if P > 0:
            return x_padded[:, :, P:-P, P:-P]
        
        return x_padded

    def forward(self, x):
        self.x_shape = x.shape

        ### Convert to Matmul with Im2Col ###
        self.x_cols, out_h, out_w = self.im2col(x)
        B, n_patches, C_K_K = self.x_cols.shape

        ### Flatten for Matmul in Linear Layer ###
        x_cols_flat = self.x_cols.reshape(B * n_patches, C_K_K)

        ### Pass through Linear Layer ###
        out_cols_flat = self.conv_linear.forward(x_cols_flat)

        # Reshape back to (B, out_channels, out_h, out_w)
        out_cols = out_cols_flat.reshape(B, n_patches, self.out_channels)
        out = out_cols.transpose(0, 2, 1).reshape(B, self.out_channels, out_h, out_w)

        return out

    def backward(self, grad_out):
        B, C, H, W = self.x_shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        ### Compute Output Shape ###
        out_h = (H + 2*P - K)//S + 1
        out_w = (W + 2*P - K)//S + 1

        ### Reshape grad_out to (B*n_patches, out_channels) so its in the form that ###
        ### nn.Linear expects our gradients to be in ###
        n_patches = out_h * out_w
        grad_cols = grad_out.reshape(B, self.out_channels, -1).transpose(0, 2, 1)
        grad_cols_flat = grad_cols.reshape(B * n_patches, self.out_channels)

        ### Compute Gradients with our Linear Layer ###
        grad_x_cols_flat = self.conv_linear.backward(grad_cols_flat)

        ### Save the gradients in this layer reshaping to our original weights shape ####
        self.grad_w = self.conv_linear.weight.grad.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.grad_b = self.conv_linear.bias.grad

        ### Reshape back to (B, n_patches, C*K*K) in the form the convolution expects it ###
        grad_x_cols = grad_x_cols_flat.reshape(B, n_patches, self.in_channels * K * K)

        ### Convert back to original image shape ###
        grad_x = self.col2im(grad_x_cols, self.x_shape, out_h, out_w)

        return grad_x

class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # By default, layers are in training mode

    def forward(self, x):
        if self.training:
            # Create mask of zeros and ones
            self.mask = (np.random.rand(*x.shape) >= self.p) 
            ### Scaling so mask divides all non-masked values by 1/p to maintain
            ### the overall expected value of the tensor 
            self.mask = self.mask / (1.0 - self.p)
            return x * self.mask
        else:
            return x  # No dropout in evaluation

    def backward(self, output_grad):
        if self.training:
            return output_grad * self.mask
        else:
            return output_grad

class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        ### Learnable Parameters that allow for scale and shift of the activations ###
        self.gamma = GradTensor(np.ones((1, num_features, 1, 1)))
        self.beta = GradTensor(np.zeros((1, num_features, 1, 1)))

        ### Running Buffers of Estimates ###
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.training = True

    def forward(self, x):
        self.x = x

        if self.training:
            ### Average over everything but the channels ###
            mean = x.mean(axis=(0,2,3), keepdims=True)
            var = x.var(axis=(0,2,3), keepdims=True)

            ### Save For Backward Pass ###
            self.mean = mean
            self.var = var

            ### Normalize ###
            self.x_hat = (x - mean) / np.sqrt(var + self.eps)

            ### Update Running States (with momentum) ###
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        ### Scale and Shift with Gamma/Beta ###
        out = self.gamma.params * self.x_hat + self.beta.params

        return out
    
    def backward(self, output_grad):
        N, C, H, W = self.x.shape

        ### Update Gamma and Beta ###
        ### dL/d_gamma = d_L/d_out * d_out/d_gamma 
        self.gamma.grad = np.sum(output_grad * self.x_hat, axis=(0,2,3), keepdims=True)
        self.beta.grad = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)

        ### gradient d_l/d_x ###
        ### d_l/d_x = d_l/d_x_hat * d_x_hat / d_x
        # Gradient w.r.t. input
        dx_hat = output_grad * self.gamma.params
        var_eps = self.var + self.eps

        dx = (1. / np.sqrt(var_eps)) * (
            dx_hat - np.mean(dx_hat, axis=(0,2,3), keepdims=True)
            - self.x_hat * np.mean(dx_hat * self.x_hat, axis=(0,2,3), keepdims=True)
        )

        return dx

class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class NeuralNetwork:
    """
    The most basic Neural Network Ever!
    """
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)

    def parameters(self):
        parameters = []
        for layer in self.layers:
            # General Layer
            if hasattr(layer, 'weight'):
                parameters.append(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                parameters.append(layer.bias)
            
            # Conv2d layer which uses a linear internally
            if hasattr(layer, 'conv_linear'):
                if hasattr(layer.conv_linear, 'weight'):
                    parameters.append(layer.conv_linear.weight)
                if hasattr(layer.conv_linear, 'bias') and layer.conv_linear.bias is not None:
                    parameters.append(layer.conv_linear.bias)
            
            # BatchNorm layer
            if hasattr(layer, 'gamma'):
                parameters.append(layer.gamma)
            if hasattr(layer, 'beta'):
                parameters.append(layer.beta)
            
            # Composite layers like TransformerLayer
            if hasattr(layer, 'parameters'):
                parameters.extend(layer.parameters())

        return parameters
    
    def named_parameters(self):
        named_params = []
        for idx, layer in enumerate(self.layers):
            layer_name = f"layer_{idx}_{layer.__class__.__name__}"
            
            # Linear layer
            if hasattr(layer, 'weight'):
                named_params.append((f"{layer_name}.weight", layer.weight))
            if hasattr(layer, 'bias') and layer.bias is not None:
                named_params.append((f"{layer_name}.bias", layer.bias))
            
            # Conv2d layer
            if hasattr(layer, 'conv_linear'):
                if hasattr(layer.conv_linear, 'weight'):
                    named_params.append((f"{layer_name}.conv_linear.weight", layer.conv_linear.weight))
                if hasattr(layer.conv_linear, 'bias') and layer.conv_linear.bias is not None:
                    named_params.append((f"{layer_name}.conv_linear.bias", layer.conv_linear.bias))
            
            # BatchNorm layer
            if hasattr(layer, 'gamma'):
                named_params.append((f"{layer_name}.gamma", layer.gamma))
            if hasattr(layer, 'beta'):
                named_params.append((f"{layer_name}.beta", layer.beta))
            
        return named_params
    
    def train(self):
        """Set network to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """Set network to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def __repr__(self):
        model_repr = "NeuralNetwork(\n"
        
        for layer in self.layers:
            if isinstance(layer, Linear):
                model_repr += f"  Linear(in_features={layer.in_features}, out_features={layer.out_features}, bias={layer.bias is not None})\n"
            elif isinstance(layer, Sigmoid):
                model_repr += "  Sigmoid()\n"
            elif isinstance(layer, ReLU):
                model_repr += "  ReLU()\n"
            elif isinstance(layer, SLOW_SoftMax): 
                model_repr += " SLOW_Softmax()\n"
            elif isinstance(layer, SoftMax):
                model_repr += "  SoftMax()\n"
            elif isinstance(layer, Flatten):
                model_repr += "  Flatten()\n"
            elif isinstance(layer, Conv2d):
                model_repr += (f"  Conv2D(in_channels={layer.in_channels}, out_channels={layer.out_channels}, "
                               f"kernel_size={layer.kernel_size}, stride={layer.stride}, padding={layer.padding})\n")
        model_repr += ")"
        return model_repr
    