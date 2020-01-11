#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
# theano specific
import theano
import theano.tensor as T


# In[2]:


# creamos dataset
trX = np.linspace(-1, 1, 101)

trY = np.linspace(-1, 1, 101)

for i in range(len(trY)):
    trY[i] = math.log(1 + 0.5 * abs(trX[i])) + trX[i] / 3 + np.random.randn() * 0.033


# In[3]:


# measure data length
m = len(trX)
# create placeholders for our data
x = T.vector('x')
y = T.vector('y')

# CREATE THEANO PARAMS
w0 = np.array([0.0]) # initialize fitting parameters
w0_theano = theano.shared(w0,name='w0')

w1 = np.array([0.0]) # initialize fitting parameters
w1_theano = theano.shared(w1,name="w1")


prediction = T.log(1 + w0_theano*T.abs_(x)) + w1_theano*x

cost = T.sum(T.pow(prediction-y,2))/(2*m)

grad_w0 = T.grad(cost,w0_theano)
grad_w1 = T.grad(cost,w1_theano)

# Some gradient descent settings
iterations = 500
alpha = 0.05

train = theano.function([x,y],cost,updates = [(w0_theano, w0_theano-alpha*grad_w0), 
                                              (w1_theano, w1_theano-alpha*grad_w1)])
# test = theano.function([x],prediction)

for i in range(iterations):
    for j in range(m):
        costM = train(np.array([trX[j]]), np.array([trY[j]]))
    # costM = train(trX, trY)
    if i==0:
        print("Cost(%i):" % i)
        print(costM)
        
print("\nCost(%i):" % i)
print(costM)


# In[4]:


# despues de entrena, cogemos los coeficientes
w0_coef, w1_coef =  w0_theano.get_value()[0], w1_theano.get_value()[0]
print("\nCoeficiente para W0:", w0_coef)
print("\nCoeficiente para W1:", w1_coef)


# In[5]:


def model(x, w0=w0_coef, w1=w1_coef):
    return np.log(1 + w0*np.abs(x)) + w1*x


# In[6]:


# creamos visualizaciones para ver el funcionamiento de nuestro modelo
plt.title("Data")
plt.plot(trX, trY, "r.")
plt.plot(trX, model(trX), "b-")
plt.legend(["Data points", "regression model"])
plt.show()

