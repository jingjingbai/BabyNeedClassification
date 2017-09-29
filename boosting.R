library(ggplot2)
library(dplyr)
library(glm2)
library(mlbench)
library(reshape2)

#load dataset1 from: http://archive.ics.uci.edu/ml/datasets/Iris
df1 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
names(df1) = c('Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class')

#load dataset2
df2 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"), header = TRUE)
names(df2) = c('Recency', 'Frequency', 'Monetary', 'Time', 'Donation')
#factorize the classification result column
df2$Class = as.factor(df2$Donation)
df2$Donation = NULL

#boosting codes
library(caret)

#define a function to compute the prediction accuracy on both testing and training data sets with various inputs
vBoosting = function(dataset, partition = seq(0.05, 0.95, by = 0.05), repetition = 10)
{
  accuracy_rep = NULL
  accuracy_all = NULL
  
  set.seed(100)
  
  #slice the original data set into training and testing data sets at various % of the total observations
  for (j in partition)
  {
    #repeat each kNN classification and average to reduce noise
    for (i in 1:repetition)
    {
      df = dataset
      
      #slice the original data set into training and testing data sets at various % of the total observations
      df_training = sample_frac(df, size = j, replace = FALSE)
      df_testing = setdiff(df, df_training)
      
      #obtain prediction accuracy on the testing set
      fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)
      model = train(Class ~., data = df_training, method = 'gbm', trControl = fitControl, verbose = F)
      

      fit_testing = predict(model, df_testing, type = 'raw')
      accuracy_testing = mean(ifelse(fit_testing == df_testing[, length(df)], 1, 0))

      #obtain prediction accuracy on the training set
      fit_training = predict(model, df_training, type = 'raw')
      accuracy_training = mean(ifelse(fit_training == df_training[, length(df)], 1, 0))
      
      #storage data
      accuracy_rep = rbind(accuracy_rep, c(accuracy_testing, accuracy_training))
    }
    
    #average all repetition
    accuracy_all = rbind(accuracy_all, apply(accuracy_rep, 2, mean))
  }
  
  accuracy_all = as.data.frame(cbind(partition, accuracy_all))
  names(accuracy_all) = c('training_size', 'testing', 'training')
  return(accuracy_all)
}

#Iris dataset
#plot prediction accuracy for both testing and training sets 
boosting_accuracy_df1 = NULL
#start the clock
ptm = proc.time()
boosting_accuracy_df1 = vBoosting(dataset = df1, partition = seq(0.4, 0.95, by = 0.05), repetition = 1)
#stop the clock
boosting_time1 = proc.time() - ptm
boosting_accuracy_df1 = melt(boosting_accuracy_df1, id = 'training_size')
names(boosting_accuracy_df1) = c('training_size', 'dataset', 'accuracy')
ggplot(boosting_accuracy_df1, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) Decision Tree Boosting accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))



#Transfusion dataset
#plot prediction accuracy for both testing and training sets 
boosting_accuracy_df2 = NULL
#start the clock
ptm = proc.time()
boosting_accuracy_df2 = vBoosting(dataset = df2, partition = seq(0.4, 0.95, by = 0.05), repetition = 1)
#stop the clock
boosting_time2 = proc.time() - ptm
boosting_accuracy_df2 = melt(boosting_accuracy_df2, id = 'training_size')
names(boosting_accuracy_df2) = c('training_size', 'dataset', 'accuracy')
ggplot(boosting_accuracy_df2, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) Decision Tree Boosting accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))
