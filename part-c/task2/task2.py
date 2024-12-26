from EDF import *



a=Conv(np.array([[2,0],[-1,0]]),np.array([[1,0,2],[-2,0,0],[0,2,-1]]))
b=MaxPooling(np.array([[1,0,2],[-2,0,0],[0,2,-1]]),(2,2),1)
a.forward()
b.forward()
print(a.value)
print(b.value)