# Predictions on whether Weight Lifting Exercises were done correctly
seemingli  
23 August 2015  
#
#### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we would use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information on the Weight Lifting Exercise Dataset is available from the website here: http://groupware.les.inf.puc-rio.br/har [^1]. We would then predict the manner in which they did the exercise by building a random forest model and use the prediction model to predict 20 different test cases. 

# 
#### Data & R Packages

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test cases are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

For this project, the following R packages will be used:

* dplyr

* caret

* ggplot2

* randomForest

* doParallel


#
#### Pre-processing of Data
As there were 160 variables in the dataset, we looked at the data to see if there were variables we could remove without affecting prediction accuracy. We noticed many columns with >90% NAs or blanks which were likely to be summary indicators, e.g. variables which start with "min". We also noticed that the first 7 columns were identifiers which were not necessary in the training model, e.g. "user_name". Removing these columns, we were left with 53 variables in the dataset. 

```r
# Read training data
df<-read.csv("pml-training.csv")
# sum no. of NAs or blanks in columns
colSums(is.na(df)|df=="") 
```

```
##                        X                user_name     raw_timestamp_part_1 
##                        0                        0                        0 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                        0                        0                        0 
##               num_window                roll_belt               pitch_belt 
##                        0                        0                        0 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                        0                        0                    19216 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                    19216                    19216                    19216 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                    19216                    19216                    19216 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                    19216                    19216                    19216 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                    19216                    19216                    19216 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                    19216                    19216                    19216 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                    19216                    19216                    19216 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                    19216                    19216                    19216 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                    19216                    19216                    19216 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                        0                        0                        0 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                        0                        0                        0 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                        0                        0                        0 
##                 roll_arm                pitch_arm                  yaw_arm 
##                        0                        0                        0 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                        0                    19216                    19216 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                    19216                    19216                    19216 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                    19216                    19216                    19216 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                    19216                    19216                        0 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                        0                        0                        0 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                        0                        0                        0 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                        0                        0                    19216 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                    19216                    19216                    19216 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                    19216                    19216                    19216 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                    19216                    19216                    19216 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                    19216                    19216                    19216 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                    19216                    19216                        0 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                        0                        0                    19216 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                    19216                    19216                    19216 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                    19216                    19216                    19216 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                    19216                    19216                    19216 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                    19216                    19216                    19216 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                    19216                    19216                        0 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                    19216                    19216                    19216 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                    19216                    19216                    19216 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                    19216                    19216                    19216 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                    19216                        0                        0 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                        0                        0                        0 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                        0                        0                        0 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                        0                        0                        0 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                        0                    19216                    19216 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                    19216                    19216                    19216 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                    19216                    19216                    19216 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                    19216                    19216                    19216 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                    19216                    19216                    19216 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                    19216                        0                    19216 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                    19216                    19216                    19216 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                    19216                    19216                    19216 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                    19216                    19216                    19216 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                        0                        0                        0 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                        0                        0                        0 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                        0                        0                        0 
##                   classe 
##                        0
```

```r
# remove columns with many NAs or blanks
df2<-select(df, -matches('^min|max|avg|stddev|var|amplitude|avg|skewness|kurtosis')) 
# remove identifier columns
df3<-df2[,8:60]
dim(df3)
```

```
## [1] 19622    53
```
#
#### Creation of training and test sets 
The data is then split into 60% training & 40% test sets using the createDataPartition function of the caret package.

```r
inTrain<-createDataPartition(y=df3$classe,p=0.6,list=FALSE)
training<-df3[inTrain,]
test<-df3[-inTrain,]
```
#
#### Random Forest Prediction Model using caret package
The prediction model will be built using Random Forest in the caret package train function as Random Forest gives relatively high accuracy even though the training time may be longer than other classifier models. Due to speed considerations, we would use 5-folds cross-validation resampling method instead of the caret defaults.

```r
my_model_file <- "my_model_file_v04.Rds"
set.seed(543)
if (file.exists(my_model_file)) {
  # Read the model in and assign it to a variable.
  modFit <- readRDS(my_model_file)
} else {
  # Otherwise, run the training
  modFit <- train(classe~ .,data=training,method="rf",trControl=trainControl(method="cv",number=5),
                  prox=TRUE,allowParallel=TRUE)
}
modFit
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9422, 9420, 9422, 9420, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9888753  0.9859274  0.001182077  0.001494245
##   27    0.9864974  0.9829180  0.003041182  0.003847341
##   52    0.9787700  0.9731401  0.004804381  0.006080861
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
From the above, we could see that caret has automatically picked 2 predictors (mtry=2) to split the trees as it was the most accurate (98.9%) under cross validation. Caret had also tried the split by 27 and 52 predicators and for the training set, the accuracy were also high at 98.6% and 97.9% respectively. As long as the training set isn't too fundamentally different from the test set, we should expect that our accuracy on the test set should be around 98% as well.


```r
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.87%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    4    1    0    0 0.001493429
## B   17 2253    9    0    0 0.011408513
## C    0   20 2031    3    0 0.011197663
## D    0    0   39 1889    2 0.021243523
## E    0    0    2    6 2157 0.003695150
```
By calling the finalModel as above, we can see the confusion matrix which shows the classification errors and that the in sample error rate is 0.87%. The error rate by number of predictors and the top 30 predictors by variable importance have also been plotted below.

```r
plot(modFit,main="Accuracy by number of Randomly Selected Predictors")
```

![](Project_files/figure-html/unnamed-chunk-6-1.png) 

```r
plot(varImp(modFit),top=30,main="Top 30 Predictors by Variable Importance")
```

![](Project_files/figure-html/unnamed-chunk-6-2.png) 

#
#### Cross Validation of the Prediction Model
Next, we run the prediction on the remaining 40% test dataset using the predict function in the caret package. We've compiled a table below on whether the observations in the test dataset were classified correctly.

```r
predtest <- predict(modFit,test)
test$predRight <- predtest==test$classe
table(predtest,test$classe)
```

```
##         
## predtest    A    B    C    D    E
##        A 2230    3    0    0    0
##        B    2 1514    8    0    0
##        C    0    1 1357   10    0
##        D    0    0    3 1276    1
##        E    0    0    0    0 1441
```

```r
missClass = function(values, prediction) {
    sum(prediction != values)/length(values)
}
errRate = round(missClass(test$classe, predtest)*100,3)
```
As we can see, only a small number of obsevations were misclassified & the expected out of sample error with the test dataset is 0.357%.

#
#### Prediction on Test Cases
Lastly, we run the prediction on the 20 test cases to obtain text files that will be used for submission.

```r
testing=read.csv("pml-testing.csv")
pred <- predict(modFit,testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    if (i < 10 ) filename = paste0("problem_id_","0",i,".txt")
    else  filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
```
[^1]: 
*Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.*
