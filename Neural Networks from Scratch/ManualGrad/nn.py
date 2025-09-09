import numpy as np
import math 

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
class GradTensor:

    def __init__(self, params):
        self.params = params
        self.grad = None
    
    def _zero_grad(self):
        self.grad = None
    
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
        if bias is not None:
            self.bias = GradTensor(
                np.random.uniform(
                    low=-math.sqrt(k), 
                    high=math.sqrt(k), 
                    size=(1,out_features)
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

        ### Deriviate w.r.t. W: X^t @ output_grad 
        self.weight.grad = self.x.T @ output_grad

        ### Derivative of Bias is jsut the output grad summed along batch 
        self.bias.grad = output_grad.sum(axis=0, keepdims=True)

        ### We need derivative w.r.t for the next step 
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

class SoftMax:

    """
    Softmax normalization to convert logits -> probabilities
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
            
            # Create the Jacobian Matrix -S_iS_j for non-diagonal terms, S_i (1-S_i) for diagonal terms
            j = -sample_probs.reshape(-1,1) * sample_probs.reshape(1,-1)
            j[np.diag_indices(j.shape[0])] = sample_probs * (1 - sample_probs)

            a = output_grad[i] @ j
            
            gradient[i] = a
        
        return gradient

class _CrossEntropyLoss:
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
            if isinstance(layer, Linear):
                parameters.append(layer.weight)
                if layer.bias is not None:
                    parameters.append(layer.bias)
        return parameters
    
    def __repr__(self):
        # Start by printing the class name for the whole model
        model_repr = "NeuralNetwork(\n"
        
        # Print each layer's class name and parameters
        for layer in self.layers:
            if isinstance(layer, Linear):
                model_repr += f"  Linear(in_features={layer.in_features}, out_features={layer.out_features}, bias={layer.bias is not None})\n"
            elif isinstance(layer, Sigmoid):
                model_repr += "  Sigmoid()\n"
            elif isinstance(layer, ReLU):
                model_repr += "  ReLU()\n"
            elif isinstance(layer, SoftMax):
                model_repr += "  SoftMax()\n"
        
        # Close the model string representation
        model_repr += ")"
        
        return model_repr