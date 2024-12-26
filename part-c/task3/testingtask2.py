import numpy as np
from EDF import *
x_node = Input()
conv1 = FastConv(np.random.randn(16, 3, 3, 3)*0.01, x_node)
relu1 = ReLU(conv1)
pool1 = MaxPooling(relu1, (2, 2), stride=2)
conv2 = FastConv(np.random.randn(32, 16, 3, 3)*0.01, pool1)
relu2 = ReLU(conv2)
pool2 = MaxPooling(relu2, (2, 2), stride=2)
conv3 = FastConv(np.random.randn(64, 32, 3, 3)*0.01, pool2)
relu3 = ReLU(conv3)
pool3 = MaxPooling(relu3, (2, 2), stride=2)
conv4 = FastConv(np.random.randn(128, 64, 3, 3)*0.01, pool3)
relu4 = ReLU(conv4)
pool4 = MaxPooling(relu4, (2, 2), stride=2)
flatten = Flatten(pool4)
fc = Linear(flatten, out_features=10, in_features=512)

graph = []
visited = set()

def topo_sort(node):
    if node in visited:
        return
    visited.add(node)
    for inp in node.inputs:
        topo_sort(inp)
    graph.append(node)

topo_sort(fc)

test_input = np.random.randn(1, 3, 64, 64)
x_node.value = test_input

for node in graph:
    node.forward()

print(fc.value.shape)
print(fc.value)
