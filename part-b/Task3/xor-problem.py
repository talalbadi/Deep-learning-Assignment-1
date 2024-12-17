import numpy as np
from EDF import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

TOTAL_SAPLES = 200

LEARNING_RATE = 0.02
EPOCHS = 1000
TEST_SIZE = 0.25

#Define the means
MEANS = np.array([[0, 0],[0,10],[10,0],[10,10]])
COV = np.array([[[1.5, 1], [1, 1.5]],
                [[1.5, 0.5], [0.5, 1.5]],[[1, 0], [0, 1]],
                [[1, 0], [0, 1]]])
SAMPLES_PER_CLASS=int(TOTAL_SAPLES*0.5)

CA1,CA2 = multivariate_normal.rvs(MEANS[3], COV[3], int(TOTAL_SAPLES*0.25)),multivariate_normal.rvs(MEANS[0], COV[3], int(TOTAL_SAPLES*0.25))
CB1,CB2 = multivariate_normal.rvs(MEANS[2], COV[3], int(TOTAL_SAPLES*0.25)),multivariate_normal.rvs(MEANS[1], COV[3], int(TOTAL_SAPLES*0.25))
WIDTH=20
# Combine the points and generate labels
X = np.vstack((np.vstack((CA1,CA2)), np.vstack((CB1,CB2))))
y = np.hstack((np.zeros(SAMPLES_PER_CLASS), np.ones(SAMPLES_PER_CLASS)))

# Plot the generated data
plt.scatter([X[:, 0]], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
BATCH_SIZE=16

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# ModModel parameters
n_features = X_train.shape[1]
n_output = 1
x_node = Input()
y_node = Input()

# Build computation graph
h1_node = Linear(x_node,WIDTH,n_features)
activatedH1 = Sigmoid(h1_node)

h2_node = Linear(activatedH1,WIDTH,WIDTH)
activatedH2 = Sigmoid(h2_node)#output layer
O_node = Linear(activatedH2,n_output,WIDTH)
activatedOutput = Sigmoid(O_node)
loss = BCE(y_node, activatedOutput)

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
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t][0]

# Training loop
epochs = 1000
learning_rate = 0.01
for epoch in range(epochs):
    loss_value = 0
    for i in range(int(X_train.shape[0]/BATCH_SIZE)):
        k=i*BATCH_SIZE

        x_node.value=X_train[k:k+BATCH_SIZE]
        #x1_node.value = X_train[i][0].reshape(1, -1)
       # x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = y_train[k:k+BATCH_SIZE].reshape(-1, 1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
for i in range(X_test.shape[0]):
  
    #x1_node.value = X_test[i][0].reshape(1, -1)
    x_node.value=X_test[i]
    #x2_node.value = X_test[i][1].reshape(1, -1)
    forward_pass(graph)
    if np.round(activatedOutput.value)== y_test[i]:
        correct_predictions += 1
    #the above method count the the boundary as a correct precdition!
    #that's why i used the below method
    #if activatedOutput.value >0.5 and y_test[i]==1 or activatedOutput.value <0.5 and y_test[i]==0:
     #correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([i, j]).reshape(1, -1)
    forward_pass(graph)
    Z.append(activatedOutput.value)

Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Decision Boundary for XOR - batch size {BATCH_SIZE} ,\n EPHOC {epoch}, width {WIDTH}, learning rate {learning_rate}')
plt.savefig(f'Decision Boundary for XOR  batch size {BATCH_SIZE} EPHOC {epoch} learning rate {learning_rate}.png')
plt.show()
