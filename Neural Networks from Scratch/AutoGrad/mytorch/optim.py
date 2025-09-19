import cupy as cp

class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):

        for param in self.parameters:
            if param.requires_grad:
                param.data -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = 0

class Adam:

    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8):

        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ### Create Momentum Vector for Each Parameter ###
        self.m = [cp.zeros_like(p.data) for p in parameters if p.requires_grad]
        self.v = [cp.zeros_like(p.data) for p in parameters if p.requires_grad]

        ### Step Index for Bias Correction ###
        self.t = 0
        
    def step(self):

        self.t += 1
        for i, param in enumerate(self.parameters):

            if param.requires_grad:
            
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

                bias_corrected_m = self.m[i] / (1 - self.beta1**self.t)
                bias_corrected_v = self.v[i] / (1 - self.beta2**self.t)
                
                param.data -= self.lr * bias_corrected_m / (cp.sqrt(bias_corrected_v) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = 0