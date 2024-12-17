import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]

class Linear(Node):
    def __init__(self,   x , columnsNO,RowsNO):
        A= Parameter(np.random.randn(columnsNO, RowsNO)* 0.01)
        b= Parameter(np.random.randn(columnsNO))
        super().__init__([A,x,b])

    def forward(self):
       a,x,b=self.inputs
       self.value=(x.value@a.value.T)+( b.value)
    def backward(self):
        A,x,b =self.inputs
        self.gradients[A]=self.outputs[0].gradients[self].T@x.value
        self.gradients[x]=self.outputs[0].gradients[self]@A.value
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=0)



class Softmax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        x = self.inputs[0].value
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.value = exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def backward(self):
        dout = self.outputs[0].gradients[self]  
        batch_size, num_classes = self.value.shape
        dX = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            y = self.value[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dX[i] = dout[i].dot(jacobian)
        self.gradients[self.inputs[0]] = dX
 
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value*np.log(y_pred.value+10e-10)-(1-y_true.value)*np.log(1-y_pred.value+10e-10))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value)+10e-10)
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value+10e-10) - np.log(1-y_pred.value+10e-10))
class CrossEntropy(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = -np.sum(y_true.value * np.log(y_pred.value +  1e-11)) / y_true.value.shape[0]

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (-y_true.value / (y_pred.value +  1e-11)) / y_true.value.shape[0]
        self.gradients[y_true] = np.zeros_like(y_true.value)
