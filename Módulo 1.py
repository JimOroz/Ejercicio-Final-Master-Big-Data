#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Ejercicio 1

import json

sentimiento = open("sentimientos.txt")
valores = {}
for linea in sentimiento:
    termino, valor = linea.split("\t")
    valores[termino] = int(valor)
tweets = open("salida_tweets.txt")
for linea in tweets:
    total = 0
    linea = linea.strip()
    if not linea:
      continue     
    data = json.loads(linea)
    if "text" not in linea:
      continue     
    for sentimiento, valor in valores.items():
        if sentimiento in data["text"]:
            total += valor
    print("- EL SIGUIENTE TWEET: '{}', TIENE UN SENTIMIENTO ASOCIADO DE: {}".format(data["text"], total))


# In[2]:


#Ejercicio 2.1

import json

sentimiento = open("sentimientos.txt")
valores = {}
for linea in sentimiento:
    termino, valor = linea.split("\t")
    valores[termino] = int(valor)
tweets = open("salida_tweets.txt")
for linea in tweets:
    tweet = json.loads(linea)
    if 'text' in tweet:
        palabras = tweet['text'].split(' ')
        suma = 0
        for palabra in palabras:
            if palabra in valores:
                suma +=valores[palabra]
        for palabra in palabras:
            if palabra not in valores:
                valores[palabra]=suma
                print (palabra + ': ' + str(suma))
                
        
 


# In[4]:


#Ejercicio 2.2

import json

sentimiento = open("sentimientos.txt")
valores = {}
for linea in sentimiento:
    termino, valor = linea.split("\t")
    valores[termino] = int(valor)
tweets = open("salida_tweets.txt")
for linea in tweets:
    tweet = json.loads(linea)
    if 'text' in tweet:
        palabras = tweet['text'].split(' ')
        suma = 0
        media = 0
        for palabra in palabras:
            if palabra in valores:
                suma +=valores[palabra]
                media = suma / len(tweet['text'].split(' '))
        for palabra in palabras:
            if palabra not in valores:
                valores[palabra]=suma
                print (palabra + ': ' + str(media))

