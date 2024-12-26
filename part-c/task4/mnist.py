import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from EDF import *

(train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()

train_X = train_X.reshape((-1, 1, 28, 28)).astype(np.float32) / 255.0
test_X  = test_X.reshape((-1, 1, 28, 28)).astype(np.float32) / 255.0

n_output = 10

onehot_y = np.zeros((train_y.shape[0], n_output), dtype=np.float32)
onehot_y[np.arange(train_y.shape[0]), train_y] = 1

onehot_ytest = np.zeros((test_y.shape[0], n_output), dtype=np.float32)
onehot_ytest[np.arange(test_y.shape[0]), test_y] = 1

x_node = Input()
y_node = Input()

conv1_weights = np.random.randn(16, 1, 3, 3).astype(np.float32) * np.sqrt(2/(1*3*3))
conv1 = FastConv(conv1_weights, x_node)
relu1 = ReLU(conv1)
pool1 = MaxPooling(relu1, (2, 2), stride=2)

conv2_weights = np.random.randn(32, 16, 3, 3).astype(np.float32) * np.sqrt(2/(16*3*3))
conv2 = FastConv(conv2_weights, pool1)
relu2 = ReLU(conv2)
pool2 = MaxPooling(relu2, (1, 1), stride=2)

conv3_weights = np.random.randn(64, 32, 3, 3).astype(np.float32) * np.sqrt(2/(32*3*3))
conv3 = FastConv(conv3_weights, pool2)
relu3 = ReLU(conv3)
pool3 = MaxPooling(relu3, (1, 1), stride=1)

conv4_weights = np.random.randn(128, 64, 3, 3).astype(np.float32) * np.sqrt(2/(64*3*3))
conv4 = FastConv(conv4_weights, pool3)
relu4 = ReLU(conv4)
flatten = Flatten(relu4)

fc = Linear(flatten, out_features=10, in_features=512)

softmax = Softmax(fc)

loss = CrossEntropy(y_node, softmax)

graph = []
trainable = []
visited = set()

def topo_sort(node):
    if node in visited:
        return
    visited.add(node)
    for inp in node.inputs:
        topo_sort(inp)
    graph.append(node)
    if isinstance(node, Parameter):
        trainable.append(node)

topo_sort(loss)

learning_rate = 0.01
EPOCHS = 1
BATCH_SIZE = 64

def sgd_update(trainables, LEARNING_RATE):
    for t in trainables:
        t.value -= LEARNING_RATE * t.gradients[t]

for epoch in range(EPOCHS):
    total_loss = 0
    n_batches = train_X.shape[0] // BATCH_SIZE

    idxs = np.random.permutation(train_X.shape[0])
    train_X = train_X[idxs]
    onehot_y = onehot_y[idxs]

    for i in range(n_batches):
        x_batch = train_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = onehot_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        x_node.value = x_batch
        y_node.value = y_batch

        for node in graph:
            node.forward()

        for node in reversed(graph):
            node.backward()

        sgd_update(trainable, learning_rate)

        total_loss += loss.value

    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.3f}")

correct = 0
conf_matrix = np.zeros((n_output, n_output), dtype=int)

for i in range(test_X.shape[0]):
    x_node.value = test_X[i:i+1]
    y_node.value = onehot_ytest[i:i+1]

    for node in graph:
        node.forward()

    pred = np.argmax(softmax.value, axis=1)[0]
    true = np.argmax(onehot_ytest[i])
    if pred == true:
        correct += 1
    conf_matrix[true, pred] += 1

acc = correct / test_X.shape[0]
print(f"\nTest accuracy = {acc*100:.2f}%")

plt.figure(figsize=(6,6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
