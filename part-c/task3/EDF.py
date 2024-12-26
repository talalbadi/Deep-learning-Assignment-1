import numpy as np

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

class Input(Node):
    def __init__(self):
        super().__init__()
    def forward(self, value=None):
        if value is not None:
            self.value = value
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self):
        pass
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class FastConv(Node):
    def __init__(self, kernel, input_node):
        kernel_node = Parameter(kernel)
        super().__init__([kernel_node, input_node])
        self.windows_2d = None
        self.x_shape = None
    def forward(self):
        kernel = self.inputs[0].value
        x = self.inputs[1].value
        self.x_shape = x.shape
        B, in_ch, H, W = x.shape
        out_ch, _, kh, kw = kernel.shape
        out_H = H - kh + 1
        out_W = W - kw + 1
        windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
        windows = windows.transpose(0, 2, 3, 1, 4, 5)
        windows_2d = windows.reshape(-1, in_ch * kh * kw)
        kernel_2d = kernel.reshape(out_ch, -1)
        out_2d = windows_2d @ kernel_2d.T
        out_4d = out_2d.reshape(B, out_H, out_W, out_ch)
        self.value = out_4d.transpose(0, 3, 1, 2)
        self.windows_2d = windows_2d
    def backward(self):
        kernel = self.inputs[0].value
        B, in_ch, H, W = self.x_shape
        out_ch, _, kh, kw = kernel.shape
        dout = sum([n.gradients[self] for n in self.outputs])
        out_H, out_W = dout.shape[2], dout.shape[3]
        dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)
        dkernel_2d = dout_2d.T @ self.windows_2d
        dkernel = dkernel_2d.reshape(out_ch, in_ch, kh, kw)
        kernel_2d = kernel.reshape(out_ch, -1)
        dx_cols = dout_2d @ kernel_2d
        dx = np.zeros((B, in_ch, H, W), dtype=dout.dtype)
        idx = 0
        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    patch = dx_cols[idx].reshape(in_ch, kh, kw)
                    dx[b, :, i:i+kh, j:j+kw] += patch
                    idx += 1
        self.gradients[self.inputs[0]] = dkernel
        self.gradients[self.inputs[1]] = dx

class ReLU(Node):
    def __init__(self, input_node):
        super().__init__([input_node])
    def forward(self):
        x = self.inputs[0].value
        self.value = np.maximum(0, x)
    def backward(self):
        dout = sum([n.gradients[self] for n in self.outputs])
        dx = dout * (self.inputs[0].value > 0)
        self.gradients[self.inputs[0]] = dx

class MaxPooling(Node):
    def __init__(self, input_node, pool_size, stride):
        super().__init__([input_node])
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None
    def forward(self):
        x = self.inputs[0].value
        B, C, H, W = x.shape
        ph, pw = self.pool_size
        st = self.stride
        out_H = (H - ph) // st + 1
        out_W = (W - pw) // st + 1
        out = np.zeros((B, C, out_H, out_W), dtype=x.dtype)
        self.mask = np.zeros_like(x, dtype=bool)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * st
                w_start = j * st
                region = x[:, :, h_start:h_start+ph, w_start:w_start+pw]
                region_2d = region.reshape(B, C, -1)
                max_vals = region_2d.max(axis=2)
                out[:, :, i, j] = max_vals
                mask_2d = (region_2d == max_vals[..., None])
                mask_4d = mask_2d.reshape(B, C, ph, pw)
                self.mask[:, :, h_start:h_start+ph, w_start:w_start+pw] = mask_4d
        self.value = out
    def backward(self):
        dout = sum([n.gradients[self] for n in self.outputs])
        B, C, out_H, out_W = dout.shape
        ph, pw = self.pool_size
        st = self.stride
        dx = np.zeros_like(self.inputs[0].value)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * st
                w_start = j * st
                dx[:, :, h_start:h_start+ph, w_start:w_start+pw] += (
                    self.mask[:, :, h_start:h_start+ph, w_start:w_start+pw]
                    * dout[:, :, i, j][:, :, None, None]
                )
        self.gradients[self.inputs[0]] = dx

class Flatten(Node):
    def __init__(self, input_node):
        super().__init__([input_node])
        self.orig_shape = None
    def forward(self):
        x = self.inputs[0].value
        self.orig_shape = x.shape
        self.value = x.reshape(x.shape[0], -1)
    def backward(self):
        dout = sum([n.gradients[self] for n in self.outputs])
        self.gradients[self.inputs[0]] = dout.reshape(self.orig_shape)

class Linear(Node):
    def __init__(self, x, out_features, in_features):
        std = np.sqrt(2.0 / in_features)
        W_init = std * np.random.randn(out_features, in_features)
        b_init = np.zeros(out_features)
        W = Parameter(W_init)
        b = Parameter(b_init)
        super().__init__([W, x, b])
    def forward(self):
        W, x, b = self.inputs
        self.value = x.value @ W.value.T + b.value
    def backward(self):
        W, x, b = self.inputs
        dout = sum([n.gradients[self] for n in self.outputs])
        self.gradients[W] = dout.T @ x.value
        self.gradients[x] = dout @ W.value
        self.gradients[b] = np.sum(dout, axis=0)

class Softmax(Node):
    def __init__(self, logits_node):
        super().__init__([logits_node])
    def forward(self):
        x = self.inputs[0].value
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.value = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def backward(self):
        dout = sum([n.gradients[self] for n in self.outputs])
        B, C = self.value.shape
        dX = np.zeros_like(self.value)
        for i in range(B):
            y = self.value[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dX[i] = np.dot(dout[i], jacobian)
        self.gradients[self.inputs[0]] = dX

class CrossEntropy(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
    def forward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        self.value = -np.sum(y_true * np.log(y_pred + 1e-11)) / y_true.shape[0]
    def backward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        self.gradients[self.inputs[1]] = (-y_true / (y_pred + 1e-11)) / y_true.shape[0]
        self.gradients[self.inputs[0]] = np.zeros_like(y_true)
