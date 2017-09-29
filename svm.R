library(ggplot2)
library(dplyr)
library(glm2)
library(mlbench)
library(reshape2)
library(e1071)


#load dataset1 from: http://archive.ics.uci.edu/ml/datasets/Iris
df1 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
names(df1) = c('Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class')

df2 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"), header = TRUE)
names(df2) = c('Recency', 'Frequency', 'Monetary', 'Time', 'Donation')
#factorize the classification result column
df2$Donation = as.factor(df2$Donation)

#SVM codes

library(e1071)
vsvm = function(dataset, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, kernel = 'radial')
{
  accuracy_rep = NULL
  accuracy_all = NULL
  
  set.seed(100)
  
  for (j in partition)
  {
    for (i in 1:repetition)
    {
      df = dataset
      df_training = sample_frac(df, size = j, replace = FALSE)
      df_testing = setdiff(df, df_training)
      

      model = svm(x = df_training[, -length(df)], y = df_training[, length(df)], kernel = kernel)
      
      fit_testing = predict(model, df_testing[,-length(df)])
      accuracy_testing = mean(ifelse(fit_testing == df_testing[, length(df)], 1, 0))

      fit_training = predict(model, df_training[, -length(df)])
      accuracy_training = mean(ifelse(fit_training == df_training[, length(df)], 1, 0))
      
      accuracy_rep = rbind(accuracy_rep, c(accuracy_testing, accuracy_training))
    }
    
    accuracy_all = rbind(accuracy_all, apply(accuracy_rep, 2, mean))
  }
  
  accuracy_all = as.data.frame(cbind(partition, accuracy_all))
  names(accuracy_all) = c('training_size', 'testing', 'training')
  return(accuracy_all)
}

#Iris dataset
#plot prediction accuracy for both testing and training sets
svm_accuracy_df1 = NULL
#start the clock
ptm = proc.time()
svm_accuracy_df1 = vsvm(dataset = df1, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, kernel = 'radial')
#stop the clock
svm_time1 = proc.time() - ptm
svm_accuracy_df1 = melt(svm_accuracy_df1, id = 'training_size')
names(svm_accuracy_df1) = c('training_size', 'dataset', 'accuracy')
ggplot(svm_accuracy_df1, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) SVM Prediction accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))



#plot prediciton accuracy for both testing and training sets with 60% of the sample being training set as a function of kernels
svm_accuracy = NULL
for (i in c('radial', 'linear', 'sigmoid', 'polynomial'))
{
  svm_kernel = vsvm(dataset = df1, partition = 0.6, repetition = 10, kernel = i)
  svm_accuracy = rbind(svm_accuracy, svm_kernel[, -1])
}

svm_accuracy = cbind(c('radial', 'linear', 'sigmoid', 'polynomial'), svm_accuracy)

names(svm_accuracy) = c('kernel', 'testing', 'training')
svm_accuracy = melt(svm_accuracy, id = 'kernel')
names(svm_accuracy) = c('kernel', 'dataset', 'accuracy')
ggplot(svm_accuracy, aes(kernel, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) SVM Prediction accuracy vs. kernels") + coord_cartesian(ylim = c(0.5,1))


#Transfusion dataset
#plot prediction accuracy for both testing and training sets
svm_accuracy_df2 = NULL
#start the clock
ptm = proc.time()
svm_accuracy_df2 = vsvm(dataset = df2, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, kernel = 'radial')
#stop the clock
svm_time2 = proc.time() - ptm
svm_accuracy_df2 = melt(svm_accuracy_df2, id = 'training_size')
names(svm_accuracy_df2) = c('training_size', 'dataset', 'accuracy')
ggplot(svm_accuracy_df2, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) SVM Prediction accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))



#plot prediciton accuracy for both testing and training sets with 60% of the sample being training set as a function of kernels
svm_accuracy = NULL
for (i in c('radial', 'linear', 'sigmoid', 'polynomial'))
{
  svm_kernel = vsvm(dataset = df2, partition = 0.6, repetition = 10, kernel = i)
  svm_accuracy = rbind(svm_accuracy, svm_kernel[, -1])
}

svm_accuracy = cbind(c('radial', 'linear', 'sigmoid', 'polynomial'), svm_accuracy)

names(svm_accuracy) = c('kernel', 'testing', 'training')
svm_accuracy = melt(svm_accuracy, id = 'kernel')
names(svm_accuracy) = c('kernel', 'dataset', 'accuracy')
ggplot(svm_accuracy, aes(kernel, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) SVM Prediction accuracy vs. kernels") + coord_cartesian(ylim = c(0.5,1))

