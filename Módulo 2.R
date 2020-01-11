#1)
#Buscamos el archivo, cargamos los datos y realizamos y unas series de gráficas para su análisis visual.
#Separamos las variables que no son tan simetricas (más llamativas).

getwd()
setwd("C:/Users/antonio/Desktop")
datos<-read.csv("CASO_FINAL_crx.data",header=F, na.strings = c("NA", "?"))
datos
str(datos)
summary(datos)

boxplot(datos$V2 ~ datos$V16)
boxplot(datos$V3 ~ datos$V16)
boxplot(datos$V8 ~ datos$V16) #(separa)
boxplot(datos$V11 ~ datos$V16) #(separa)
boxplot(datos$V14 ~ datos$V16) #(separa)
boxplot(datos$V15 ~ datos$V16) #(separa)

plot(datos$V16 ~ datos$V1) 
plot(datos$V16 ~ datos$V4) #(separa)
plot(datos$V16 ~ datos$V5) #(separa)
plot(datos$V16 ~ datos$V6) #(separa)
plot(datos$V16 ~ datos$V7) #(separa)
plot(datos$V16 ~ datos$V9) #(mejor separa)
plot(datos$V16 ~ datos$V10) #(separa)
plot(datos$V16 ~ datos$V12) 
plot(datos$V16 ~ datos$V13) #(separa)


#2)
#Instalamos previamente el paquete missForest y después lo llamamos. Realizamos una nueva matrix y comprobamos con summary.
library(missForest)
?missForest
datos1 <- missForest(datos)
datos1 <- datos1$ximp
summary(datos1)
#3)
#Realizamos la división de datos1

train <- datos1[1:590,]
test <- datos1[591:nrow(datos1),]

#4)
#Instalamos el paquete previamente después realizamos una matrix numérica de train y test. Realizamaos los modelos ridge alpha=0 y lasso alpha=1.
#Dibujamos las graficass de AUC y elegimos las. 

library("glmnet")
library("caret")
train <- data.matrix(train)
test <- data.matrix(test)


# Modelo ridge
modelo_ridge<- cv.glmnet(train[,1:15], as.factor(train[,16]), family="binomial",  alpha=0, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(modelo_ridge)
modelo_ridge$lambda.min #Mejor lambda
min(modelo_ridge$cvm) #Este es el error de lambda
coef(modelo_ridge, s=modelo_ridge$lambda.min) # la mayoría de los coeficientes son casi nulos

#Modelo lasso
modelo_lasso<- cv.glmnet(train[,1:15], as.factor(train[,16]), family="binomial",  alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(modelo_lasso)
modelo_lasso$lambda.min #Mejor lambda
min(modelo_lasso$cvm) #Este es el error de lambda
coef(modelo_lasso, s=modelo_lasso$lambda.min) # la mayoría de los coeficientes son casi nulos

#Métricas

y_ridge <- predict(modelo_ridge, as.matrix(test[,1:15]), s = "lambda.min", type='class')
rid<-confusionMatrix(as.factor(y_ridge), as.factor(test[,16]), mode="everything")
rid


y_lasso <- predict(modelo_lasso, as.matrix(test[,1:15]), s = "lambda.min", type='class')
las<-confusionMatrix(as.factor(y_lasso), as.factor(test[,16]), mode="everything")
las

#5) Hacemos el log odds en el modelo con mejor auc.

coef(modelo_lasso)
exp(coef(modelo_lasso))

#6)

las$table
VP = (84*100/(100*85+6*20))*100
VP


