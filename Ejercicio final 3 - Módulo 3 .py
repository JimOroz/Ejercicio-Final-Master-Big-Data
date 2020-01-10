#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('pylab', '')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[10]:


import pandas as pd
filename = "crime_data.csv"
df = pd.read_csv(filename, sep = ',')
df.head()


# In[11]:


col_names = list(df.columns)
col_names.remove('State')

df_state = df[col_names]
df_state.head()


# In[12]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def plot_sillhouette(blobs, figure_name, max_k = 10, n_init = 10):
    sillhouette_avgs = []
    
    for k in range(2, max_k):
        kmean = KMeans(n_clusters = k, n_init = n_init).fit(blobs)
        sillhouette_avgs.append(silhouette_score(blobs, kmean.labels_))
        
    plot(range(2, max_k), sillhouette_avgs)
    title(figure_name)
    
plot_sillhouette(df_state, 'df')


# In[14]:


kmeans = KMeans(n_clusters = 2, n_init = 10).fit(df_state)
kmeans.cluster_centers_


# In[15]:


clust = kmeans.predict(df_state)

for i in range(max(clust) + 1):
    print ("Cluster", i)
    print (df["State"][clust == i])

