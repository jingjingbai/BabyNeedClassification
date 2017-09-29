library(ggplot2)
library(dplyr)
library(glm2)
library(mlbench)
library(reshape2)

#load dataset1 from: http://archive.ics.uci.edu/ml/datasets/Iris
df1 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
names(df1) = c('Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class')

#load dataset2 from: http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data
df2 = read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"), header = TRUE)
names(df2) = c('Recency', 'Frequency', 'Monetary', 'Time', 'Donation')

#kNN codes
library(class)

#define a function to compute the prediction accuracy on both testing and training data sets with various inputs
vknn = function(dataset, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, neighbors = 3)
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
      fit_testing = knn(train = df_training[, -length(df)], test = df_testing[, -length(df)], cl = df_training[, length(df)], k = neighbors)
      accuracy_testing = mean(ifelse(fit_testing == df_testing[, length(df)], 1, 0))

      #obtain prediction accuracy on the training set
      fit_training = knn(train = df_training[, -length(df)], test = df_training[, -length(df)], cl = df_training[, length(df)], k = neighbors)
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
#plot prediction accuracy for both testing and training sets with 3 neighbors
knn_accuracy_df1 = NULL
#start the clock
ptm = proc.time()
knn_accuracy_df1 = vknn(dataset = df1, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, neighbors = 10)
#stop the clock
knn_time1 = proc.time() - ptm
knn_accuracy_df1 = melt(knn_accuracy_df1, id = 'training_size')
names(knn_accuracy_df1) = c('training_size', 'dataset', 'accuracy')
ggplot(knn_accuracy_df1, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) kNN Prediction accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))

#plot prediciton accuracy for both testing and training sets with 60% of the sample being training set as a function of neighbors
knn_accuracy = NULL
for (i in seq(1.0, 50.0, by = 2.0))
{
  knn_neighbor = vknn(dataset = df1, partition = 0.6, repetition = 10, neighbors = i)
  knn_accuracy = rbind(knn_accuracy, knn_neighbor[, -1])
}

knn_accuracy = cbind(seq(2.0, 50.0, by = 2.0), knn_accuracy)

names(knn_accuracy) = c('neighbors', 'testing', 'training')
knn_accuracy = melt(knn_accuracy, id = 'neighbors')
names(knn_accuracy) = c('neighbors', 'dataset', 'accuracy')
ggplot(knn_accuracy, aes(neighbors, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) kNN Prediction accuracy vs. number of neighbors") + coord_cartesian(ylim = c(0.5,1))


#Transfusion dataset
#plot prediction accuracy for both testing and training sets with 3 neighbors
knn_accuracy_df2 = NULL
#start the clock
ptm = proc.time()
knn_accuracy_df2 = vknn(dataset = df2, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, neighbors = 10)
#stop the clock
knn_time2 = proc.time() - ptm
knn_accuracy_df2 = melt(knn_accuracy_df2, id = 'training_size')
names(knn_accuracy_df2) = c('training_size', 'dataset', 'accuracy')
ggplot(knn_accuracy_df2, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) kNN Prediction accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))

#plot prediciton accuracy for both testing and training sets with 60% of the sample being training set as a function of neighbors
knn_accuracy = NULL
for (i in seq(1.0, 50.0, by = 2.0))
{
  knn_neighbor = vknn(dataset = df2, partition = 0.6, repetition = 10, neighbors = i)
  knn_accuracy = rbind(knn_accuracy, knn_neighbor[, -1])
}

knn_accuracy = cbind(seq(2.0, 50.0, by = 2.0), knn_accuracy)

names(knn_accuracy) = c('neighbors', 'testing', 'training')
knn_accuracy = melt(knn_accuracy, id = 'neighbors')
names(knn_accuracy) = c('neighbors', 'dataset', 'accuracy')
ggplot(knn_accuracy, aes(neighbors, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) kNN Prediction accuracy vs. number of neighbors") + coord_cartesian(ylim = c(0.5,1))

