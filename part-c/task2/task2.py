from EDF import *

x=Input()
x.value=np.array([[1,0,2],[-2,0,0],[0,2,-1]])
kernal=Input()
kernal.value=np.array([[2,0],[-1,0]])

a=Conv(kernal,x)
p=Input()
p.value=1
size=Input()
size.value=(2,2)
b=MaxPooling(x,size,p)
a.forward()
b.forward()