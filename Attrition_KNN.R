#Data partitioning and modeling
library(outliers)
library(caret)
library(dplyr)

#Setting the working directory
#Setting the working directory
setwd('C:/Users/Aditya/Documents/Fall 3rd Semester/Applied Machine Learning/Project 3')

#Reading in the data
df <- read.csv('Attrition.csv')


#Dropping categorical variable with just 1 level
df = select(df, -EmployeeCount,-StandardHours,-Over18)

#Converting to categorical variables

df$Education <- as.factor(df$Education) 
df$EnvironmentSatisfaction <- as.factor(df$EnvironmentSatisfaction)
df$JobInvolvement <- as.factor(df$JobInvolvement)
df$JobLevel <- as.factor(df$JobLevel)
df$JobSatisfaction <- as.factor(df$JobSatisfaction)
df$PerformanceRating <- as.factor(df$PerformanceRating)
df$RelationshipSatisfaction <- as.factor(df$RelationshipSatisfaction)
df$StockOptionLevel <- as.factor(df$StockOptionLevel)
df$TrainingTimesLastYear <- as.factor(df$TrainingTimesLastYear)
df$WorkLifeBalance <- as.factor(df$WorkLifeBalance)

#Scaling Continous Features
library(BBmisc)
nums <- sapply(df, is.numeric)
numeric <- df[,nums]

#Finding the categorical variables
categ <- sapply(df, is.factor)
categorical <- df[,categ]


#Dummy variable conversion
X <- select(categorical,-Attrition)
dmy <- dummyVars(" ~ . ", data = X ) 
df3 <- data.frame(predict(dmy, newdata = X))
str(df3)


#combining Categorical and Numeric
df5 <- cbind.data.frame(df3, numeric)

#Scaling whole Dataset
scaled_df <- normalize(df5, method = "standardize", range = c(0, 1))

df6 <-  cbind.data.frame(scaled_df, df$Attrition)
colnames(df6)[86] <- "Attrition"

#Dividing the dataset into Train and Test

df_train <- df6[1:735, ]
df_test <- df6[736:1470, ]

#K Nearest Neighbours

set.seed(100)
ctrl <- trainControl(method="repeatedcv",repeats = 10)
knnFit <- train(Attrition ~ ., data = df_train, method = "knn", trControl = ctrl, tuneLength = 10)

#Output of kNN fit
knnFit

plot(knnFit)


knnPredict <- predict(knnFit,newdata = df_test)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, df_test$Attrition )

#Plotting Area under curve
library(AUC)

plot(roc(knnPredict,df_test$Attrition))

auc(roc(knnPredict,df_test$Attrition))

