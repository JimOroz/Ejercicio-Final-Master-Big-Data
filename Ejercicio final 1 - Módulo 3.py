#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


# In[2]:


filename = "auto.csv"

# leemos el archivo
df = pd.read_csv(filename, sep = ',')
df.head()


# In[3]:


# Separación de la variable objetivo y las explicativas
target = 'mpg'
features = [col for col in df.columns if col != target]


# In[4]:


# Listado de variables disponibles para hacer un modelo.
for var in features:
    print(var , ':' , len(set(df[var])))


# In[5]:


pd.plotting.scatter_matrix(df, figsize = (12, 12), diagonal = 'kde');


# In[6]:


# creamos un modelo y le pasamos todos los datos
regresor = DecisionTreeRegressor(max_depth=5)

regresor.fit(df[features], df[target])


# In[7]:


preds = regresor.predict(df[features])


# In[8]:


# creamos una visualizacioón de las predicciones y los datos reales
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title("Datos reales (rojo) vs predicciones (azul)")
plt.plot(preds, "b.")
plt.plot(df[target], "r.")
plt.legend(["predicciones del modelo", "datos reales"], loc="upper left")

plt.subplot(1,2,2)
plt.title("Error cometido por punto predicho (menor es mejor)")
plt.plot(preds-df[target], "r.")
plt.legend(["Error (y-y_pred)"], loc="upper left")

plt.show()

