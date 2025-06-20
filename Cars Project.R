install.packages("readr")
install.packages("corrplot")

mydata=read.csv("D:/GreatLakes/Machine Learning/Cars/cars.csv", header =TRUE)
variable.names(mydata)
str(mydata)
dim(mydata)
view(mydata)
summary(mydata)

sapply(mydata,function(x)sum(is.na(x)))
mydata$`MBA`[is.na(mydata$`MBA`)]=median(mydata$`MBA`,na.rm=T)
sapply(mydata,function(x)sum(is.na(x)))

library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(corrplot)
library (caret)
## Loading required package: lattice## Loading required package: ggplot2
library (car)
## Loading required package: carData
library (DMwR)
library(readr)
library(DMwR)



cor.mydata = cor(mydata[-c(2,9)],)
round(cor.mydata,2)
corrplot(cor.mydata)
corrplot(cor.mydata,method = "number",number.cex = .6)


boxplot(mydata$Age)
quantile(mydata$Age,probs = seq(0,1,0.05))
subset1=mydata[which(mydata$Age>37),]  # Capping in the upper value
subset1$Age=37
mydata[which(mydata$Age>37),]=subset1

boxplot(mydata$Work.Exp)
quantile(mydata$Work.Exp,probs = seq(0,1,0.05))
subset2=mydata[which(mydata$Work.Exp>15),]  # Capping in the upper value
subset2$Work.Exp=15
mydata[which(mydata$Work.Exp>15),]=subset2

boxplot(mydata$Salary)
quantile(mydata$Salary,probs = seq(0,1,0.05))
subset3=mydata[which(mydata$Salary>24),]  # Capping in the upper value
subset3$Salary=24
mydata[which(mydata$Salary>24),]=subset3

boxplot(mydata$Distance)
quantile(mydata$Distance,probs = seq(0,1,0.05))
subset4=mydata[which(mydata$Distance>18),]  # Capping in the upper value
subset4$Distance=18
mydata[which(mydata$Distance>18),]=subset4



#Bivariate Analysis
boxplot(mydata$Age~mydata$Engineer, main = "Age vs Engineer")
boxplot(mydata$Age~mydata$MBA, main = "Age vs MBA")

boxplot(mydata$Salary~mydata$Engineer, main = "Salary vs Engineer")
boxplot(mydata$Salary~mydata$MBA, main = "Salary vs MBA")

# Salary vs Transport Model

boxplot(mydata$Salary~mydata$Transport, main="Salary vs Transport")
cor(mydata$Age,mydata$Work.Exp)


# Age vs Transport Model

boxplot(mydata$Age~mydata$Transport, main="Age vs Transport")

boxplot(mydata$Distance~mydata$Transport, main="Distance vs Transport")



table(mydata$Gender, mydata$Transport)

hist(mydata$Work.Exp,col="blue",main="Distribution of work Experience")

table (mydata$license,mydata$Transport)
boxplot(mydata$Work.Exp~mydata$Gender)


mydata$Engineer=as.factor(mydata$Engineer)
mydata$MBA=as.factor(mydata$MBA)
mydata$license=as.factor(mydata$license)



mydata%>%keep(is.numeric)%>%gather()%>%ggplot(aes(value))+facet_wrap(~key,scales="free")+
  geom_histogram(col="Blue")


# Hypothesis Testing


#Preparation of the data
carsbasedata=mydata
str(carsbasedata)
carsbasedata<-knnImputation(carsbasedata)

carsbasedata$CarUsage<-ifelse(carsbasedata$Transport =='Car',1,0)
table(carsbasedata$CarUsage)


sum(carsbasedata$CarUsage ==1)/nrow(carsbasedata)

carsbasedata$CarUsage<-as.factor(carsbasedata$CarUsage)


#Model Building and Data Split

set.seed(400)
carindex<-createDataPartition(carsbasedata$CarUsage, p=0.7,list = FALSE,times = 1)
carsdatatrain<-carsbasedata[carindex,]
carsdatatest<-carsbasedata[-carindex,]
prop.table(table(carsdatatrain$CarUsage))


prop.table(table(carsdatatest$CarUsage))


carsdatatrain<-carsdatatrain[,c(1:8,10)]
carsdatatest<-carsdatatest[,c(1:8,10)]

## The train and test data have almost same percentage of cars usage as the base data
## Apply SMOTE on Training data set


attach(carsdatatrain)
carsdataSMOTE<-SMOTE(CarUsage~., carsdatatrain, perc.over = 250,perc.under = 150)
prop.table(table(carsdataSMOTE$CarUsage))


outcomevar<-'CarUsage'
regressors<-c("Age","Work.Exp","Salary","Distance","license","Engineer","MBA","Gender")
trainctrl<-trainControl(method = 'repeatedcv',number = 10,repeats = 3)
carsglm<-train(carsdataSMOTE[,regressors],carsdataSMOTE[,outcomevar],method = "glm", family = "binomial",trControl = trainctrl)

summary(carsglm$finalModel)

carglmcoeff<-exp(coef(carsglm$finalModel))
write.csv(carglmcoeff,file = "Coeffs.csv")
varImp(object = carsglm)

plot(varImp(object = carsglm), main="Vairable Importance for Logistic Regression")

## Model Interpretation

carusageprediction<-predict.train(object = carsglm,carsdatatest[,regressors],type = "raw")
confusionMatrix(carusageprediction,carsdatatest[,outcomevar], positive='1')


carusagepreddata<-carsdatatest
carusagepreddata$predictusage<-carusageprediction


# Improving the model

trainctrlgn<-trainControl(method = 'cv',number = 10,returnResamp = 'none')
carsglmnet<-train(CarUsage~Age+Work.Exp+Salary+Distance+license, data = carsdataSMOTE, method = 'glmnet', trControl = trainctrlgn)
carsglmnet

varImp(object = carsglmnet)
plot(varImp(object = carsglmnet), main="Vairable Importance for Logistic Regression - Post Ridge Regularization")

# Prediction using the regularized model
carusagepredictiong<-predict.train(object = carsglmnet,carsdatatest[,regressors],type = "raw")
confusionMatrix(carusagepredictiong,carsdatatest[,outcomevar], positive='1')



# Inference & Prediction Using Linear Discriminant Analysis

carsbasedatalda<-read.csv("D:/GreatLakes/Machine Learning/Cars/cars.csv", header = TRUE)
carsbasedatalda$Gender<-as.factor(carsbasedatalda$Gender)
carsbasedatalda$Engineer<-as.factor(carsbasedatalda$Engineer)
carsbasedatalda$MBA<-as.factor(carsbasedatalda$MBA)
carsbasedatalda<-knnImputation(carsbasedatalda)
set.seed(400)
carindexlda<-createDataPartition(carsbasedatalda$Transport, p=0.7,list = FALSE,times = 1)
carstrainlda<-carsbasedatalda[carindexlda,]
carstestlda<-carsbasedatalda[-carindexlda,]
carstrainlda$license<-as.factor(carstrainlda$license)
carstestlda$license<-as.factor(carstestlda$license)
cartrainlda.car<-carstrainlda[carstrainlda$Transport %in% c("Car", "Public Transport"),]
cartrainlda.twlr<-carstrainlda[carstrainlda$Transport %in% c("2Wheeler", "Public Transport"),]
cartrainlda.car$Transport<-as.character(cartrainlda.car$Transport)
cartrainlda.car$Transport<-as.factor(cartrainlda.car$Transport)
cartrainlda.twlr$Transport<-as.character(cartrainlda.twlr$Transport)
cartrainlda.twlr$Transport<-as.factor(cartrainlda.twlr$Transport)
prop.table(table(cartrainlda.car$Transport))


prop.table(table(cartrainlda.twlr$Transport))

carldatwlrsm <- SMOTE(Transport~., data = cartrainlda.twlr, perc.over = 150, perc.under=200)
table(carldatwlrsm$Transport)

carldacarsm <- SMOTE(Transport~., data = cartrainlda.car, perc.over = 175, perc.under=200)
table(carldacarsm$Transport)

carldacar<-carldacarsm[carldacarsm$Transport %in% c("Car"),]
carsdatatrainldasm<-rbind(carldatwlrsm,carldacar)
str(carsdatatrainldasm)

table(carsdatatrainldasm$Transport)

attach(carsdatatrainldasm)

trainctrllda<-trainControl(method = 'cv',number = 10)
carslda<-train(Transport~Age+Work.Exp+Salary+Distance+license+Gender+Engineer+MBA ,data = carsdatatrainldasm, method="lda", trControl=trainctrllda)
carslda$finalModel

plot(varImp(object = carslda),main="Variable Importance for Linear Discriminant Analysis" )


carsldapredict<-predict.train(object = carslda,newdata = carstestlda)
confusionMatrix(carsldapredict,carstestlda[,9])





#Improve LDA Model by Regularization

trainctrlpda<-trainControl(method = 'cv',number = 10, returnResamp = 'all')

carspda<-train(Transport~Age+Work.Exp+Salary+Distance+license+Gender+Engineer+MBA ,data = carsdatatrainldasm, method="pda", trControl=trainctrlpda)
carspda$finalModel

carspda

plot(varImp(object = carspda), main="Variable Importance for Penalized Discriminant Analysis")


carspdapredict<-predict.train(object = carspda,newdata = carstestlda)
confusionMatrix(carspdapredict,carstestlda[,9])



#Prediction using CART

carscart<-train(Transport~.,carsdatatrainldasm,method = 'rpart', trControl = trainControl(method = 'cv',number = 5,savePredictions = 'final'))
carscart$finalModel

library(rattle)
fancyRpartPlot(carscart$finalModel)

predictions_CART<-predict(carscart,carstestlda)
confusionMatrix(predictions_CART,carstestlda$Transport)


#Prediction using Boosting

boostcontrol <- trainControl(number=10)

xgbGrid <- expand.grid(
  eta = 0.3,
  max_depth = 1,
  nrounds = 50,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1, subsample = 1
)

carsxgb <-  train(Transport ~ .,carsdatatrainldasm,trControl = boostcontrol,tuneGrid = xgbGrid,metric = "Accuracy",method = "xgbTree")

carsxgb$finalModel

# Predict using Test Dataset

predictions_xgb<-predict(carsxgb,carstestlda)
confusionMatrix(predictions_xgb,carstestlda$Transport)




# Prediction Using Multinomial Logistic Regression

carsmlr<-train(Transport ~.,carsdatatrainldasm,method = "multinom")
carsmlr$finalModel

carmlrcoeff<-exp(coef(carsmlr$finalModel))
write.csv(carmlrcoeff,file = "Coeffsmlr.csv")
plot(varImp(object=carsmlr), main = "Variable Importance for Multinomial Logit")

# Predict using the test data

predictions_mlr<-predict(carsmlr,carstestlda)
confusionMatrix(predictions_mlr,carstestlda$Transport)


# Prediction using Random Forest

rftrcontrol<-control <- trainControl(method="repeatedcv", number=10, repeats=3)
mtry<-sqrt(ncol(carsdatatrainldasm))
tunegridrf <- expand.grid(.mtry=mtry)
carsrf<-train(Transport ~.,carsdatatrainldasm,method = "rf", trControl=rftrcontrol, tuneGrid = tunegridrf)
carsrf$finalModel

plot(varImp(object=carsrf), main = "Variable Importance for Random Forest")

# Predict for test data

predictions_rf<-predict(carsrf,carstestlda)
confusionMatrix(predictions_rf,carstestlda$Transport)

