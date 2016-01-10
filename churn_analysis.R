library(glmnet)
library(boot)
library(caret)
library(popbio)
library(pROC)
library(rpart)
library(rattle)
library(dplyr)

# load data
churndata <- read.csv('/Users/youluqi/webdata/churn.csv')

# data cleaning
churndata_1 <- select(churndata, -Phone,-State)
churndata_1$Int.l.Plan <- ifelse(churndata_1$Int.l.Plan=='no',0,1)
churndata_1$VMail.Plan <- ifelse(churndata_1$VMail.Plan=='no',0,1)
x <- as.matrix(churndata_1[,1:18])
y <- as.matrix(churndata_1[,19])

#######################
# Logistic Regression #
#######################

# Extremely efficient procedures for fitting lasso for logistic regression model.
LASSO_model <- cv.glmnet(x,y,family="binomial",type.measure="deviance")
plot(LASSO_model)

#largest value of lambda such that error is within 1 standard error of the minimum
LASSO_model$lambda.1se

# a fitted logistic regression for the full data.
final_model <- LASSO_model$glmnet.fit
model.coef <- coef(LASSO_model$glmnet.fit,s=LASSO_model$lambda.1se)
LASSO_pre <- predict(final_model,newx = x, s=LASSO_model$lambda.1se,type = 'response')
LASSO_pre_class <- predict(final_model,newx = x, s=LASSO_model$lambda.1se,type = 'class')


# visualization of regression results
churndata_1$Churn1 = ifelse(churndata_1$Churn.=='True.',1,0)
popbio::logi.hist.plot(churndata_1$Day.Charge, churndata_1$Churn1, boxp=FALSE,type="hist",col="gray")


# model evaluation
table(y,LASSO_pre_class)
confusionMatrix(y,LASSO_pre_class,positive = 'False.')

preobs <- data.frame(prob=LASSO_pre, obs=churndata_1$Churn1)
head(preobs)

# draw ROC plot
modelroc = roc(preobs$obs, preobs$X1)
plot(modelroc, print.auc=TRUE, auc.polygon=T, grid=c(0.1,0.2),grid.col=c('green','red'),max.auc.polygon=TRUE, auc.polygon.col="skyblue", print.thres=TRUE)


# The function createDataPartition can be used to create balanced splits of the data.
# create a single 80\20% split of the churn data:
#set.seed(123)
#trainIndex <- createDataPartition(churndata_2$Churn.,p=0.8,list=F,times = 1)
#traindata <- churndata_2[trainIndex,]
#head(traindata)
#testdata <- churndata_2[-trainIndex,]
#head(testdata)


#######################
# decision tree model #
#######################
churndata_2 <- select(churndata_1,-Churn1)
Tree_model <- rpart(Churn.~., data = churndata_2)
summary(Tree_model)

# Create decision tree
fancyRpartPlot(Tree_model)

printcp(Tree_model)
# the optimal cp value associated with the minimum error.
Tree_model$cptable[which.min(Tree_model$cptable[,"xerror"]),"CP"]
plotcp(Tree_model)

# Prediction and Model Evalution
# Confusion Matrix
Tree_pre <- predict(Tree_model, type = 'class')
table(churndata_2$Churn.,Tree_pre)
confusionMatrix(churndata_2$Churn.,Tree_pre,positive = 'False.')

# RUC
Tree_pre2 <- predict(Tree_model, type = 'prob')
preobs2 <- data.frame(prob=Tree_pre2[,1], obs=churndata_1$Churn1)
head(preobs2)
modelroc2 = roc(preobs2$obs, preobs2$prob)
plot(modelroc2, print.auc=TRUE, auc.polygon=T, grid=c(0.1,0.2),grid.col=c('green','red'),max.auc.polygon=TRUE, auc.polygon.col="skyblue", print.thres=TRUE)
