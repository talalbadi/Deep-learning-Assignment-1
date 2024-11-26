import numpy as np
from EDF import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

TOTAL_SAPLES = 200

LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

#Define the means
MEANS = np.array([[3, 3],[3,9],[9,3],[9,9]])
COV = np.array([[[1, 0.5], [0.5, 0.8]],[[1, 0], [0, 0.5]],[[0.8, 1], [0, 1]],[[0.6, 0], [0.3, 1]]])
SAMPLES_PER_CLASS=int(TOTAL_SAPLES*0.5)

CA1,CA2 = multivariate_normal.rvs(MEANS[3], COV[1], int(TOTAL_SAPLES*0.25)),multivariate_normal.rvs(MEANS[0], COV[0], int(TOTAL_SAPLES*0.25))
CB1,CB2 = multivariate_normal.rvs(MEANS[2], COV[2], int(TOTAL_SAPLES*0.25)),multivariate_normal.rvs(MEANS[1], COV[3], int(TOTAL_SAPLES*0.25))

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
batch_size=1
test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros(1)
A=np.random.randn(n_output,n_features)*0.1
b=np.random.randn(1)*0.1
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

# Create nodes
#x1_node = Input()
x_node = Input()
#x2_node = Input()
y_node = Input()

w0_node = Parameter(W0)
w1_node = Parameter(W1)
w2_node = Parameter(W2)
#b1_node = Parameter(b1)
A_node=Parameter(A)
b_node=Parameter(b)
# Build computation graph
z_node = Linear(A_node,x_node,b_node)
#u1_node = Multiply(x1_node,w1_node)
#u2_node = Multiply(x2_node,w2_node)
#u12_node = Addition(u1_node,u2_node)
#u_node = Addition(u12_node, w0_node)
sigmoid = Sigmoid(z_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
#graph = [x1_node,x2_node,w0_node,w1_node,w2_node,u1_node,u2_node,u12_node,u_node,sigmoid,loss]
graph = [x_node,A_node,b_node,z_node,y_node,sigmoid,loss]
#trainable = [w0_node,w1_node,w2_node]
trainable = [A_node,b_node]

# Training loop
epochs = 100
learning_rate = 0.001

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


for epoch in range(epochs):
    loss_value = 0
    for i in range(int(X_train.shape[0]/batch_size)):
        k=i*batch_size

        x_node.value=X_train[k:k+batch_size]
        #x1_node.value = X_train[i][0].reshape(1, -1)
       # x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = y_train[k:k+batch_size].reshape(-1, 1)

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
   # if np.round(sigmoid.value)== y_test[i]:
    #    correct_predictions += 1
    #the above method count the the boundary as a correct precdition!
    #that's why i used the below method
    if sigmoid.value >0.5 and y_test[i]==1 or sigmoid.value <0.5 and y_test[i]==0:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    #x1_node.value = np.array([i]).reshape(1, -1)
   

   # x2_node.value = np.array([j]).reshape(1, -1)
    x_node.value = np.array([i,j]) 
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Decision Boundary - batch size {batch_size}')
#plt.savefig(f'run using batch {batch_size}')
plt.show()


