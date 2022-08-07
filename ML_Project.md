---
title: "A Predictive Model to describe the manner in which certain weight lifting exercises are completed." 
author: "Dawit Aberra" 
date: August 2022 
output:
  html_document:
    keep_md: yes
---



# Synopsis

We will use a [Weight Lifting Exercise Dataset](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) in which `training` and `testing` datasets of accelerometers on the belt, forearm, arm, and dumbell of 6 participants were provided. These participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Our goal is to build a model that predicts the manner in which the participants did the exercise, with acceptable accuracy and minimal out of sample error. This is the `classe` variable in the training set which consists of six levels describing the manners. We use most of the other variables to predict with. This report describes how we built the model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases.

# Getting and Cleaning Data

Download Data. We get the training and testing data from links which are [available here](http://groupware.les.inf.puc-rio.br/har).


```r
library(caret)
training_link<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_link<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./data")) {dir.create("./data")}
if (!file.exists("./data/pml-training.csv")) 
  {download.file(training_link, destfile="./data/pml-training.csv", method="curl")}
if (!file.exists("./data/pml-testing.csv")) 
  {download.file(testing_link, destfile="./data/pml-testing.csv", method="curl")}
training <-read.csv("./data/pml-training.csv")
testing <- read.csv("./data/pml-testing.csv")
```

The `training` data contains 19622 rows and 160 columns. The `testing` data contains 20 rows and 160 columns.


```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

Remove columns with missing values or mostly not available (NA)


```r
cs <- colSums(is.na(training))
training <- training[ ,cs<0.05*ncol(training)]
testing <- testing[ , cs<0.05*ncol(training)]
```

Remove columns irrelevant to the outcome. 

```r
head(names(training), n=10)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"
```
Inspecting the column names and importance, we find that the names that contain "X", "user_name" and "_timestamp_" are pretty much irrelevant to the manner in which the exercises are performed. We suggest that the first 5 variables be removed.



```r
training <- training[,-c(1:5)] 
testing <- testing[,-c(1:5)] 
```

Remove variables with zero and near-zero variance. Data come sometimes with predictors that take a unique value across sample. Such predictor is more common. This kind of predictor is not only non-informative, it can break some models `[2].` We decided to remove variables with near zero variance.


```r
nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing <- testing[,-nzv]
```

The cleaned `training` data contains 19622 rows and 54 columns. The cleaned `testing` data contains 20 rows and 54 columns.


```r
dim(training)
```

```
## [1] 19622    54
```

```r
dim(testing)
```

```
## [1] 20 54
```

# Generate `training` and `validation` Data.

At this stage, we leave the cleaned `testing` data aside for later use only, but we split the cleaned `training` data to `training` and `validation` dataset to be used in the next few steps. We will keep 70% of the training data for training our model(s) and the rest for validation.


```r
set.seed(1000)
inTrain <- createDataPartition(y=training$classe, p=0.70, list=F)
training <- training[inTrain, ]
validation <- training[-inTrain, ]
```


```r
dim(training)
```

```
## [1] 13737    54
```

```r
dim(validation)
```

```
## [1] 4104   54
```

# Cross-Validation

Here we create a control to train the model using 5-Folds Cross Validation.


```r
tc <- trainControl(method="cv", number=5, verboseIter=F)
```

# Building and Selection of model

Here we build/train a model and view the properties: a `RandomForest` model


```r
model_rf<-train(classe ~ ., data=training, method="rf", trControl=tc)
print(model_rf$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    5 2651    1    1    0 0.0026335591
## C    0    3 2393    0    0 0.0012520868
## D    0    0    9 2243    0 0.0039964476
## E    0    1    0    5 2519 0.0023762376
```

# Use `model_rf` to predict `classe` on the `validation` data and examine accuracy and out-of-sample errors with ConFusionMatrix


```r
predictions_on_validation<-predict(model_rf, newdata=validation)
confusionMatrix(factor(validation$classe), predictions_on_validation)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1200    0    0    0    0
##          B    0  767    0    0    0
##          C    0    0  724    0    0
##          D    0    0    0  686    0
##          E    0    0    0    0  727
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2924     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2924   0.1869   0.1764   0.1672   0.1771
## Detection Rate         0.2924   0.1869   0.1764   0.1672   0.1771
## Detection Prevalence   0.2924   0.1869   0.1764   0.1672   0.1771
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

From the above output, this model is expected to predict with `100%` accuracy, and zero percent out of sample error.
I chose to keep this `RandomForrest` model and not to try another model.

# Re-train the model.

Now that we have already chosen our model to be a 'RandomForest', it is a good practice to re-train the model with the whole original training set (without spiting to validation) before using the model to predict on un-seen data (in real time applications). However, with this remark to users, we will continue with the one we already trained. 

# Use `model_rf` to predict on `testing` data. Write out the predictions for 20 test cases`.


```r
predictions_on_testing <- predict(model_rf, newdata=testing)
predictions_on_testing
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

# References:
1. [Weight Lifting Exercise Dataset](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)
2.  Near-zero variance predictors. Should we remove them? [R-Bloggers, March 16, 2014](https://www.r-bloggers.com/2014/03/near-zero-variance-predictors-should-we-remove-them/)


