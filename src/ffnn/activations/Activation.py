import numpy as np

# NB: upstream = hasil chain derivation

class Activation:
    """Base class (meant to be ABC)"""
    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, ...)
            
        Returns:
            Output array of same shape
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def backward(self, upstream_grad):
        """ Gradient calculation
        
        Args:
            upstream_grad: Gradient from next layer of shape (batch_size, ...)
            
        Returns:
            Gradient with respect to input of same shape
        """
        raise NotImplementedError("Subclasses must implement backward()")

class Linear(Activation):
    def forward(self, x):
        self.input = x
        self.output = x
        return self.output
    
    def backward(self, upstream_grad): # upstream * dx/dx = upstream * 1
        return upstream_grad

class ReLU(Activation):
    """Rectified Linear Unit"""
    def forward(self, x): # no negative
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, upstream_grad):
        mask = (self.input > 0).astype(float) # 0 or 1
        return upstream_grad * mask

class Sigmoid(Activation):
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, upstream_grad):
        return upstream_grad * self.output * (1 - self.output)

class Tanh(Activation):
    """Hyperbolic tangent"""
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, upstream_grad): # 1 - tanh^2
        return upstream_grad * (1 - self.output ** 2)

class Softmax(Activation):
    """For multi-class classification"""
    def forward(self, x):
        """
        gabisa lgsg implement; e^[some big number] bakal error
        softmax behaviour: hasilnya ga berubah kalau semua input dikurangin konstanta yang sama.
        implementasi: kurangin sama angka terbesar, baru kalkulasi outputnya
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.input = x
        return self.output
    
    def backward(self, upstream_grad):
        """Use softmax Jacobian"""
        s = self.output
        ds = upstream_grad * s
        grad_input = ds - s * np.sum(ds, axis=1, keepdims=True)
        return grad_input