#!/usr/bin/env python
# coding: utf-8

# # Crea un conjunto de datos utilizando el siguiente código:
# 
# trX = np.linspace(-1, 1, 101)
# 
# trY = np.linspace(-1, 1, 101)
# 
# for i in range(len(trY)):
# 
# trY[i] = math.log(1 + 0.5 * abs(trX[i])) + trX[i] / 3 + np.random.randn() * 0.033
# 
# # Ahora, utiliza Theano para obtener los parámetros w_0 y w_1 del siguiente modelo:
# 
# # y= log (1+w0|x|)+w1x
# 
# # utilizando los datos generados anteriormente.

# In[ ]:


import theano
import theano.tensor as T


# In[2]:


trX = np.linspace(-1, 1, 101)

trY = np.linspace(-1, 1, 101)

X = T.matrix()
Y = T.matrix()

def model(X, w, c):
    return X * w + c

w = theano.shared(np.asarray(0., dtype = theano.config.floatX))
c = theano.shared(np.asarray(0., dtype = theano.config.floatX))
y = model(X, w, c)

cost     = T.mean(T.sqr(y - Y))
gradient_w = T.grad(cost = cost, wrt = w)
gradient_c = T.grad(cost = cost, wrt = c)
updates  = [[w, w - gradient_w * 0.01], [c, c - gradient_c * 0.01]]

train = theano.function(inputs = [X, Y], outputs = cost, updates = updates)

for i in range(15):
    
    for i in range(len(trY)):

    trY[i] = math.log(1 + 0.5 * abs(trX[i])) + trX[i] / 3 + np.random.randn() * 0.033


# In[ ]:




