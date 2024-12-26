import numpy as np
from EDF import *
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import multivariate_normal

from keras import datasets

from matplotlib import pyplot
 
 
 
 
(train_X, train_y), (test_X, test_y) =  datasets.mnist.load_data()




print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()



LEARNING_RATE = 0.1
EPOCHS = 10
TEST_SIZE = 0.4
BATCH_SIZE=10
WIDTH=250
n_output = 10
columnsNO=train_X.shape[1]*train_X.shape[2]
train_X,test_X=train_X.reshape((train_X.shape[0],columnsNO)),test_X.reshape((test_X.shape[0],columnsNO))
n_features = train_X.shape[1]

onehot_y,onehot_ytest=np.zeros((train_y.shape[0],n_output)),np.zeros((test_y.shape[0],n_output))
onehot_y[np.arange(onehot_y.shape[0]),train_y]=1
onehot_ytest[np.arange(onehot_ytest.shape[0]),test_y]=1


x_node = Input()
y_node = Input()

# Build computation graph
h1_node = Linear(x_node,WIDTH,n_features)
activatedH1 = Sigmoid(h1_node)
h2_node = Linear(activatedH1,WIDTH,WIDTH)
activatedH2 = Sigmoid(h2_node)
ho_node = Linear(activatedH2,10,WIDTH)
activatedOutput = Softmax(ho_node)
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

learning_rate = 0.01
for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(int(train_X.shape[0]/BATCH_SIZE)):
        k=i*BATCH_SIZE

        x_node.value=train_X[k:k+BATCH_SIZE]
        #x1_node.value = X_train[i][0].reshape(1, -1)
       # x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = onehot_y[k:k+BATCH_SIZE]


        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / train_X.shape[0]}")

# Evaluate the model
correct_predictions = 0
confusionMatrix=np.zeros((n_output,n_output))

for i in range(test_X.shape[0]):
    x_node.value = test_X[i:i+1]
    y_node.value = test_y[i:i+1]
    forward_pass(graph)
    pred_class = np.argmax(activatedOutput.value, axis=1)[0]
    if  onehot_ytest[i,pred_class]==1:
        correct_predictions += 1
    confusionMatrix[np.where( onehot_ytest[i]==1)[0],pred_class]+=1
    


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
accuracy = correct_predictions / test_X.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusionMatrix)
