library(e1071)
#install.packages("naivebayes")
library(naivebayes)
library(rattle)
library(rpart)
library(caret)

setwd("C:/Users/gagd2/Desktop/Syracuse/IST_707/Datasets from async/Week7_NB")

filename="Kaggle-digit-train.csv"
DigitDF <- read.csv(filename, header = TRUE, stringsAsFactors = TRUE)
(head(DigitDF))
(str(DigitDF))
(nrow(DigitDF))
DigitDF$label <- as.factor(DigitDF$label)

total <- sum(is.na(DigitDF))
cat("\nThe num of missing values in the dataset is ", total)

## Create a test and train set using cross validation of picking up every 3rd row and use it as the test set.
(every3_rows<-seq(1,nrow(DigitDF),3))
DigitDF_Test=DigitDF[every3_rows, ]
DigitDF_Train=DigitDF[-every3_rows, ]
## View the created Test and Train sets
(head(DigitDF_Train))
(nrow(DigitDF_Test))
nrow(DigitDF_Train)
(table(DigitDF_Test$Label))

## Make sure you take the labels out of the testing data
(head(DigitDF_Test))
DigitDF_Test_noLabel<-DigitDF_Test[-c(1)]
DigitDF_Test_justLabel<-DigitDF_Test$label
(head(DigitDF_Test_noLabel))

####Naive Bayes model using package e1071
## formula is label ~ x1 + x2 + .  NOTE that label ~. is "use all to create model"
DigitDF_NB_train<-naiveBayes(label~., data=DigitDF_Train, na.action = na.pass)
DigitDF_NB_pred <- predict(DigitDF_NB_train, DigitDF_Test_noLabel)
summary(DigitDF_NB_train)
table(DigitDF_NB_pred,DigitDF_Test_justLabel)

confusionMatrix(DigitDF_NB_pred, DigitDF_Test_justLabel)

#accuracy is 52%

#to build a different model using naive bayes, I'm using feature selection
#for Feature selection, one thing that we could do is to remove all those attributes which have a value
# of zero for all the rows\images.
DigitDF <- read.csv(filename, header = TRUE, stringsAsFactors = TRUE)
DigitDF$label <- as.factor(DigitDF$label)
featureSelDigitDF <- DigitDF[,colSums(DigitDF!=0)>0]
str(featureSelDigitDF)
#the number of attributes reduced to 709 from 785

# We must remove other pixels with little variance through images
# *pixels with little variances will be the ones with most 0 in them

variances <- data.frame(apply(featureSelDigitDF[-1], 2, var))
colnames(variances) <- "variances"
str(variances)
plot(variances$variances, type = "l", xlab="Pixel", ylab="Pixel variance", lwd=2)

sorted_var <- variances[order(variances$variances), , drop = FALSE]
# "drop = False" allows us to maintain original row names

plot(sorted_var$variances, type = "l",xlab="Pixel", ylab="Pixel variance", lwd=2)
# there are about 200 pixels with very low variance

par(mar=c(3,9,3,9))
boxplot(variances, ylab="Pixel variance")
abline(h=4850, col="red", lwd=2) # mean value
text(5300, "Mean", font=2)
abline(h=200, col="blue", lwd=2) # where are those 200 pixels
text(600, "200 pixels", font=2)


# Seems like the first quantile includes near 200 pixels
# let's confirm that by using the 1st Qu. values from the summary
summary(variances) # 1st Qu. = 89
length(variances[variances$variances <= 89, ]) # 177 pixels, close

# Let's zoom in into the previous plot to see where does 200 pixels 
# exactly cut
plot(sorted_var$variances, type = "l", xlim = c(0,220), ylim=c(0,200))
abline(v=200) # around 150
length(variances[variances$variances <= 150, ]) # exactly 200 pixels

# Remove those pixels
less_200 <- subset(variances, variances >= 150, "variances")
pixels <- row.names(less_200)
train <- featureSelDigitDF[, c("label", pixels)] 
str(train)

### MACHINE LEARNING
# First we normalize the data
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

train_nolabel <- train[,-1]
train_nolabel_norm <- as.data.frame(lapply(train_nolabel, normalize))
train_norm <- cbind(label=train$label, train_nolabel_norm)
str(train_norm)
#let's see if the accuracy improves now
## Create a test and train set using cross validation of picking up every 3rd row and use it as the test set.
(every3_rows<-seq(1,nrow(train_norm),3))
train_norm_Test=train_norm[every3_rows, ]
train_norm_Train=train_norm[-every3_rows, ]
## View the created Test and Train sets

(nrow(train_norm_Test))

(table(train_norm_Test$Label))

## Make sure you take the labels out of the testing data

train_norm_Test_noLabel<-train_norm_Test[-c(1)]
train_norm_Test_justLabel<-train_norm_Test$label
(head(train_norm_Test_noLabel))

####Naive Bayes model using package e1071
## formula is label ~ x1 + x2 + .  NOTE that label ~. is "use all to create model"
train_norm_NB_train<-naiveBayes(label~., data=train_norm_Train, na.action = na.pass)
train_norm_NB_pred <- predict(train_norm_NB_train, train_norm_Test_noLabel)
train_norm_NB_train
table(train_norm_NB_pred,train_norm_Test_justLabel)

confusionMatrix(train_norm_NB_pred, train_norm_Test_justLabel)
#accuracy is 70% -- much better

# decision tree model

#install.packages("rattle")

DigitDF_DT <- rpart(label ~ ., train_norm_Train, method = "class")
DigitDF_DT_pred <- predict(DigitDF_DT, train_norm_Test_noLabel, type = "class")
table(DigitDF_DT_pred, train_norm_Test_justLabel)
summary(DigitDF_DT)
confusionMatrix(DigitDF_DT_pred, train_norm_Test_justLabel)
#accuracy is 60%
fancyRpartPlot(DigitDF_DT)

#Plotcp() provides a graphical representation to the cross validated error summary. 
#The cp values are plotted against the geometric mean to depict the deviation until the minimum value is reached.
plotcp(DigitDF_DT)


#Prune the tree to create an optimal decision tree
prunedDigitDF_DT<- prune(DigitDF_DT, cp= DigitDF_DT$cptable[which.min(DigitDF_DT$cptable[,"xerror"]),"CP"])
fancyRpartPlot(prunedDigitDF_DT, uniform=TRUE, main="Pruned Classification Tree")



#Decision Tree model 2 with cost parameter values of 0.05

DigitDF_DT2 <- rpart(label ~ ., train_norm_Train, method = "class", control=rpart.control(cp=0.05))
DigitDF_DT_pred2 <- predict(DigitDF_DT2, train_norm_Test_noLabel, type = "class")
table(DigitDF_DT_pred2, train_norm_Test_justLabel)
summary(DigitDF_DT2)
confusionMatrix(DigitDF_DT_pred2, train_norm_Test_justLabel)
plotcp(DigitDF_DT2)
#accuracy is 45%
fancyRpartPlot(DigitDF_DT2)

#Prune the tree to create an optimal decision tree
prunedDigitDF_DT2<- prune(DigitDF_DT2, cp= DigitDF_DT2$cptable[which.min(DigitDF_DT2$cptable[,"xerror"]),"CP"])
fancyRpartPlot(prunedDigitDF_DT2, uniform=TRUE, main="Pruned Classification Tree")

#Decision Tree model 3 with cost parameter value of 0.03
DigitDF_DT3 <- rpart(label ~ ., train_norm_Train, method = "class", control=rpart.control(cp=0.03))
DigitDF_DT_pred3 <- predict(DigitDF_DT3, train_norm_Test_noLabel, type = "class")
table(DigitDF_DT_pred3, train_norm_Test_justLabel)
summary(DigitDF_DT3)
confusionMatrix(DigitDF_DT_pred3, train_norm_Test_justLabel)
plotcp(DigitDF_DT3)
#accuracy is 53%
fancyRpartPlot(DigitDF_DT3)

#Prune the tree to create an optimal decision tree
prunedDigitDF_DT3<- prune(DigitDF_DT3, cp= DigitDF_DT3$cptable[which.min(DigitDF_DT3$cptable[,"xerror"]),"CP"])
fancyRpartPlot(prunedDigitDF_DT3, uniform=TRUE, main="Pruned Classification Tree")

#Decision Tree model 4 with cost parameter value of 0.01
DigitDF_DT4 <- rpart(label ~ ., train_norm_Train, method = "class", control=rpart.control(cp=0.01))
DigitDF_DT_pred4 <- predict(DigitDF_DT4, train_norm_Test_noLabel, type = "class")
table(DigitDF_DT_pred4, train_norm_Test_justLabel)
summary(DigitDF_DT4)
confusionMatrix(DigitDF_DT_pred4, train_norm_Test_justLabel)
plotcp(DigitDF_DT4)
#accuracy is 60%
fancyRpartPlot(DigitDF_DT4)

#Prune the tree to create an optimal decision tree
prunedDigitDF_DT4<- prune(DigitDF_DT4, cp= DigitDF_DT4$cptable[which.min(DigitDF_DT4$cptable[,"xerror"]),"CP"])
fancyRpartPlot(prunedDigitDF_DT4, uniform=TRUE, main="Pruned Classification Tree")

# apply the naive bayes and decision tree algorithms on the actual test data that did not have any labels.
### PREDICITIONS:
# In order to use our model to predict the test dataset, we need
# to pre-process the test data the same way we did with the train data

# First we remove the pixels we didn't use
filename="Kaggle-digit-test.csv"
testDF <- read.csv(filename, header = TRUE, stringsAsFactors = TRUE)
testDF$label <- as.factor(testDF$label)
reduced_testDF <- testDF[,colSums(testDF!=0)>0]

reduced_test <- reduced_testDF[,pixels]
# Then we normalize
final_test_norm <- as.data.frame(lapply(reduced_test, normalize))

# Make the predictions using naive bayes
pred <- predict(train_norm_NB_train, final_test_norm)
table(pred)

#make predictions using decision tree
pred_DT <- predict(DigitDF_DT, final_test_norm, type="class")
table(pred_DT)

pred_submit <- data.frame(ImageId=1:nrow(reduced_testDF),Label=pred)
pred_submit_DT <- data.frame(ImageId=1:nrow(reduced_testDF),Label=pred_DT)

write.table(pred_submit, "DigitRecog_submit.csv", sep=";",quote = F, row.names = F)
write.table(pred_submit_DT, "DigitRecog_DT_submit.csv", sep=";",quote = F, row.names = F)
