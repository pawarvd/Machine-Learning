
#Title: "Machine Learning Project"
##author: "Vivek Pawar"
##date: "October 20, 2018"
##output: html_document

The purpose of this project is to predict how well the dumbell biceps curl exercise was done that was monitired by various sensors.  There are five classes of how well the exercise was done. Class A corresponds to the specified execution of the exercise whereas the the other 4 (B to E) correspond to common mistakes.

*Tidyig the data*

The first step is to read the data set that is provided as training dataset and testing dataset. The training dataset has 19622 observations and 160 variables.  The testing dataset has 20 observations with 160 variables. View of data in R studio shows that there are several variables that are NAs or missing values.  In addition there are time stamps and usernames that are not useful either. 

NOTE: 
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 
```{r}
require(caret)
require(ggplot2)
require(rattle)
require(tidyverse)
require(klar)
trainingall <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
testingall <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
traindataclean <- trainingall[-c(1:7,12:36, 50:59, 69:83,87:101, 103:112, 125:139,141:150)]
testingdataclean <- testingall[-c(1:7,12:36, 50:59, 69:83,87:101, 103:112, 125:139,141:150)]
dim(traindataclean)
dim(testingdataclean)
```

So training data has 19,622 observations with 53 varaibles and testing data has 20 observations with 53 vairables. Now lets partition the training data set and use portion of that to train and test our model. So now our training dataset has 14718 variables whereas testing dataset has 4904 observations.

```{r}
inTrain <- createDataPartition(traindataclean$classe, p=0.75, list=FALSE)
traindataclean1 <- traindataclean[inTrain,]
trainvaldataclean <- traindataclean[-inTrain,]
dim(traindataclean)
dim(trainvaldataclean)
```
Since there are 53 variables, it will be difficult to look at all of them in a convinient manner. But lets look at some of the key variables such as total_acceleration for belt, arm, dumbell and forarm. The total_acce_belt seems to be different in Class A (correct way to exercise) compared to the other classes (B to E). The other total acceelrations do not show any such diference.

```{r}
par(mfrow=c(1,4))
plot(x=traindataclean1$classe, y=traindataclean1$total_accel_belt, xlab="class", ylab="Total Acce Belt")
plot(x=traindataclean1$classe, y=traindataclean1$total_accel_arm, xlab="Class", ylab="Total Acce Arm")
plot(x=traindataclean1$classe, y=traindataclean1$total_accel_dumbbell, xlab="class", ylab="Total Acce Dumbbell")
plot(x=traindataclean1$classe, y=traindataclean1$total_accel_forearm, xlab="Class", ylab="Total Acce Forearm")
```

*Modeling of the data*

As a first approach lets use classification tree method to classify the data and evaluate the model's accuracy on the subset of the data. The overall accuracy is low 0.49. 

```{r}
set.seed(1234)
trControl <- trainControl(method="cv", number=5)
rpartmodelfit <- train(classe~., data=traindataclean1, method="rpart", trControl=trControl)
fancyRpartPlot(rpartmodelfit$finalModel)
```

```{r}
predmod <- predict(rpartmodelfit, newdata=trainvaldataclean)
rpartconfM <- confusionMatrix(trainvaldataclean$classe,predmod)
rpartconfM
```

Now lets use randomforest method to classify the data and evaluate the model's accuracy on the subset of the data. The accuracy of the random forest method is 0.99 with 5 fold validation.

```{r}
set.seed(1288)
trControl <- trainControl(method="cv", number=5)
rfmodelfit <- train(classe~., data=traindataclean1, method="rf", trControl=trControl)
predmod1 <- predict(rfmodelfit, newdata=trainvaldataclean)
rfconfM <- confusionMatrix(trainvaldataclean$classe, predmod1)
rfconfM
```

Now lets try gradient boosting method. The accuracy of this is method is also 0.95

```{r}
set.seed(1288)
trControl <- trainControl(method="cv", number=5)
gbmodelfit <- train(classe~., data=traindataclean1, method="gbm", trControl=trControl,verbose=FALSE)
predmod2 <- predict(gbmodelfit, newdata=trainvaldataclean)
gbconfM <- confusionMatrix(trainvaldataclean$classe, predmod2)
gbconfM
```

Figure below shows the relative importance of top 20 variables in random forest fit and plot of accuracy vs number of predictors. Plot of top two variables is also shown 

```{r}
plot(varImp(rfmodelfit), top=20)
plot(rfmodelfit)
```

*Conclusion*
The random forest method gave the maximum accuracy (0.9933). Using that model for the validation/test data set of 20 observation gave following output.

```{r}
predmod3 <- predict(rfmodelfit, testingdataclean)
predmod3
```

