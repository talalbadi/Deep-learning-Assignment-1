import numpy as np
from EDF import *
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import multivariate_normal

TOTAL_SAPLES = 200

LEARNING_RATE = 0.1
EPOCHS = 50
TEST_SIZE = 0.4

WIDTH=64
# Combine the points and generate labels

mnist = datasets.load_digits()

X, y = mnist['data'], mnist['target'].astype(int)

# Plot the generated data


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
BATCH_SIZE=16
y_onehot = np.zeros((y.shape[0],10))
y_onehot[np.arange(y.shape[0]), y] = 1
test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]


X_train, y_train = X[train_indices], y_onehot[train_indices]
X_test, y_test = X[test_indices], y_onehot[test_indices]
# Model parameters
n_features = X_train.shape[1]
n_output = 10
x_node = Input()
y_node = Input()

# Build computation graph
h1_node = Linear(x_node,WIDTH,n_features)
activatedH1 = Sigmoid(h1_node)

h2_node = Linear(activatedH1,10,WIDTH)
activatedOutput = Softmax(h2_node)
loss = CrossEntropy(y_node, activatedOutput)

# Create graph outside the training loop
graph = []
trainable = []

def topologicalSort(node,graph,trainable):
 

    for n in node.inputs:
        topologicalSort(n,graph,trainable)
    graph.append(node)
    if isinstance(node, Parameter):
            trainable.append(node)



topologicalSort(loss,graph,trainable)
    
# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, LEARNING_RATE):
    for t in trainables:
        t.value -= LEARNING_RATE * t.gradients[t]

# Training loop
epochs = 128
learning_rate = 0.01
for epoch in range(epochs):
    loss_value = 0
    for i in range(int(X_train.shape[0]/BATCH_SIZE)):
        k=i*BATCH_SIZE

        x_node.value=X_train[k:k+BATCH_SIZE]
        #x1_node.value = X_train[i][0].reshape(1, -1)
       # x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = y_onehot[train_indices][k:k+BATCH_SIZE]


        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
confusionMatrix=np.zeros((n_output,n_output))
y_true_labels = np.argmax(y_test, axis=1)
for i in range(X_test.shape[0]):
    x_node.value = X_test[i:i+1]
    y_node.value = y_test[i:i+1]
    forward_pass(graph)
    pred_class = np.argmax(activatedOutput.value, axis=1)[0]
    if pred_class == y_true_labels[i]:
        correct_predictions += 1
    confusionMatrix[y_true_labels[i],pred_class]+=1
    


plt.figure(figsize=(8,6))
plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

tick_marks = np.arange(n_output)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = confusionMatrix.max() / 2.
for i in range(n_output):
    for j in range(n_output):
        plt.text(j, i, int(confusionMatrix[i, j]),
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusionMatrix)
