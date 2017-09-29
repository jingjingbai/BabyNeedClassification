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

#neural networks codes
library(nnet)

#define a function to compute the prediction accuracy on both testing and training data sets with various inputs
vNeural = function(dataset, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, hidden =2)
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
      
      model = nnet(Class~., data = df_training, size = hidden, rang = 0.1, decay = 5e-4, maxit = 200)
      
      fit_testing = predict(model, data = df_testing, type = 'class')
      accuracy_testing = mean(ifelse(fit_testing == df_testing[, length(df)], 1, 0))

      #obtain prediction accuracy on the training set
      fit_training = predict(model, data = df_training, type = 'class')
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
neural_accuracy_df1 = NULL
#start the clock
ptm = proc.time()
neural_accuracy_df1 = vNeural(dataset = df1, partition = seq(0.05, 0.95, by = 0.05), repetition = 10, hidden = 10)
#stop the clock
neural_time1 = proc.time() - ptm
neural_accuracy_df1 = melt(neural_accuracy_df1, id = 'training_size')
names(neural_accuracy_df1) = c('training_size', 'dataset', 'accuracy')
ggplot(neural_accuracy_df1, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) Neural Networks accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))



#Transfusion dataset
#plot prediction accuracy for both testing and training sets
neural_accuracy_df2 = NULL
#start the clock
ptm = proc.time()
neural_accuracy_df2 = vNeural(dataset = df2, partition = seq(0.05, 0.95, by = 0.05), repetition = 10)
#stop the clock
neural_time2 = proc.time() - ptm
neural_accuracy_df2 = melt(neural_accuracy_df2, id = 'training_size')
names(neural_accuracy_df2) = c('training_size', 'dataset', 'accuracy')
ggplot(neural_accuracy_df2, aes(training_size, accuracy, color = dataset)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) Neural Networks accuracy vs. training sample %") + coord_cartesian(ylim = c(0.5,1))

