#Comparing prediction accuracy for different algorithms for Iris testing dataset as a function of training sample % of total sample, with default settings.
accuracy_comparison_df1 = cbind(tree_accuracy_df1, svm_accuracy_df1[,3], knn_accuracy_df1[,3], neural_accuracy_df1[,3])
accuracy_comparison_df1 = inner_join(accuracy_comparison_df1, boosting_accuracy_df1, by = c('training_size', 'dataset'))
names(accuracy_comparison_df1) = c('training_size','dataset', 'DecisionTree', 'SVM', 'KNN', 'NeuralNetworks', 'Boosting')
accuracy_comparison_df1 = melt(accuracy_comparison_df1, id=c('training_size','dataset'))
names(accuracy_comparison_df1) = c('training_size','dataset', 'algorithm', 'accuracy')
ggplot(accuracy_comparison_df1[accuracy_comparison_df1$dataset=='testing', ], aes(training_size, accuracy, color = algorithm)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Iris Dataset) comparison of algorithms") + coord_cartesian(ylim = c(0.5,1))


###
#Comparing prediction accuracy for different algorithms for blood transfusion testing dataset as a function of training sample % of total sample, with default settings.
accuracy_comparison_df2 = cbind(tree_accuracy_df2, svm_accuracy_df2[,3], knn_accuracy_df2[,3], neural_accuracy_df2[,3])
accuracy_comparison_df2 = inner_join(accuracy_comparison_df2, boosting_accuracy_df2, by = c('training_size', 'dataset'))
names(accuracy_comparison_df2) = c('training_size','dataset', 'DecisionTree', 'SVM', 'KNN', 'NeuralNetworks', 'Boosting')
accuracy_comparison_df2 = melt(accuracy_comparison_df2, id=c('training_size','dataset'))
names(accuracy_comparison_df2) = c('training_size','dataset', 'algorithm', 'accuracy')
ggplot(accuracy_comparison_df2[accuracy_comparison_df2$dataset=='testing', ], aes(training_size, accuracy, color = algorithm)) + geom_line(size = 1) + geom_point(size =3, shape = 1) + theme(legend.position = "bottom") + ggtitle("(Transfusion Dataset) comparison of algorithms") + coord_cartesian(ylim = c(0.5,1))

###
#computation time for Iris
time_comparison_df1 = as.data.frame(cbind(tree_time1[1], svm_time1[1], knn_time1[1], neural_time1[1], boosting_time1[1]))
names(time_comparison_df1) = c('DecisionTree', 'SVM', 'KNN', 'NeuralNetworks', 'Boosting')

#computation time for blood transfusion
time_comparison_df2 = as.data.frame(cbind(tree_time2[1], svm_time2[1], knn_time2[1], neural_time2[1], boosting_time2[1]))
names(time_comparison_df2) = c('DecisionTree', 'SVM', 'KNN', 'NeuralNetworks', 'Boosting')
