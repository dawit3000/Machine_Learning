Dawit Aberra
August 2022

# A Predictive Model to describe the manner in which certain weight lifting exercises are completed.

# Synopsis

We will use a [Weight Lifting Exercise
Dataset](http://groupware.les.inf.puc-rio.br/har) in which a training
and testing datasets of accelerometers on the belt, forearm, arm, and
dumbell of 6 participants were provided. These participants were asked
to perform barbell lifts correctly and incorrectly in 5 different ways.

Our goal is to build a model that predicts the manner in which the
participants did the exercise, with acceptable accuracy and minimal out
of sample error. This is the “classe” variable in the training set which
consists of six levels describing the manners. We use most of the other
variables to predict with. This report describes how we built the model,
how we used cross validation, what we think the expected out of sample
error is, and why we made the choices we did. We will also use our
prediction model to predict 20 different test cases.

# Getting and Cleaning Data

Download Data. We get the training and testing data from links which are
[available here](http://groupware.les.inf.puc-rio.br/har).

``` r
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

The `training` dataset contains 19622 rows and 160 columns. The
`testing` dataset contains 20 rows and 160 columns.

``` r
dim(training)
```

    ## [1] 19622   160

``` r
dim(testing)
```

    ## [1]  20 160

Remove columns with missing values or mostly not available (NA)

``` r
mostlyNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, mostlyNA==F]
testing <- testing[, mostlyNA==F]
```

Remove columns irrelevant to the outcome Inspecting the variables verses
the outcome suggests that the firest 7 variables are fairly irrelevant
to the expected outcome. We decided to remove these variables.

``` r
training <- training[,-c(1:5)] 
testing <- testing[,-c(1:5)] 
```

Remove variables with zero and near-zero variance. Datasets come
sometimes with predictors that take a unique value across sample. Such
uninformative predictor is more common. This kind of predictor is not
only non-informative, it can break some models \[1\]. We decided to
remove variables with near zero variance.

``` r
nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing <- testing[,-nzv]
```

The cleaned `training` dataset contains 19622 rows and 54 columns. The
cleaned `testing` dataset contains 20 rows and 54 columns.

``` r
dim(training)
```

    ## [1] 19622    54

``` r
dim(testing)
```

    ## [1] 20 54

# Generate `training` and `validation` Data.

At this stage, we leave the cleaned `testing` data aside for later use
only, but we split the cleaned `training` dataset to `training` and
`validation` datasets to be used in the next few steps. We will keep 70%
of the training dataset for training our our model and the rest for
validation.

``` r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=F)
training <- training[inTrain, ]
validation <- training[-inTrain, ]
```

# Cross-Validation

Here we create a control to train the model using 3-Folds Cross
Validation.

``` r
tc <- trainControl(method="cv", number=3, verboseIter=F)
```

# Build a RandomForest Model and view the final parameters it chose

``` r
model_rf<-train(classe ~ ., data=training, method="rf", trControl=tc)
print(model_rf$finalModel)
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
    ## A 3905    0    0    0    1 0.0002560164
    ## B    4 2649    5    0    0 0.0033860045
    ## C    0    4 2391    1    0 0.0020868114
    ## D    0    0    9 2243    0 0.0039964476
    ## E    0    1    0    3 2521 0.0015841584

# Use `model_rf` to predict `classe` on the `validation` data and examine accuracy and out-of-sample errors with ConFusionMatrix

``` r
predictions_on_validation<-predict(model_rf, newdata=validation)
confusionMatrix(factor(validation$classe), predictions_on_validation)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1191    0    0    0    0
    ##          B    0  773    0    0    0
    ##          C    0    0  727    0    0
    ##          D    0    0    0  681    0
    ##          E    0    0    0    0  748
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9991, 1)
    ##     No Information Rate : 0.2891     
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
    ## Prevalence             0.2891   0.1876   0.1765   0.1653   0.1816
    ## Detection Rate         0.2891   0.1876   0.1765   0.1653   0.1816
    ## Detection Prevalence   0.2891   0.1876   0.1765   0.1653   0.1816
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

# Use `model_rf` to predict on `testing` dataset. Write out the predictions for 20 test cases.

``` r
predictions_on_testing <- predict(model_rf, newdata=testing)
predictions_on_testing
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

# References:

1.  Near-zero variance predicators. Should we remove them? [R-Bloggers,
    March 16,
    2014](https://www.r-bloggers.com/2014/03/near-zero-variance-predictors-should-we-remove-them/)

2.  [https://www.dataquest.io/blog/r-markdown-guide-cheatsheet/](RMarkdown%20Help)
