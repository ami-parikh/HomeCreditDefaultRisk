library('readr')


#Using imputed data
df <- read.csv('imputed_train.csv', sep=' ')

#Removing predictors which are always 0 for target


df <- subset(df, select = -c(
  
  FLAG_DOCUMENT_4,
  FLAG_DOCUMENT_10,
  FLAG_DOCUMENT_12,
  FLAG_MOBIL,
  SK_ID_CURR
  
))

#Reducing number of levels for Organization type
levels(df$ORGANIZATION_TYPE) <- list(
  Advertising=c('Advertising'),
  Agriculture=c('Agriculture'),
  Bank=c('Bank'),
  Business=c("Business Entity Type 1", "Business Entity Type 2", "Business Entity Type 3", 'Self-employed'),
  Cleaning=c('Cleaning'),
  Construction=c('Construction'),
  Culture=c('Culture'),
  Electricity=c('Electricity'),
  Emergency = c("Emergency"),
  Government = c("Government"), 
  Hotel = c("Hotel"),
  Housing = c("Housing", "Realtor"), 
  Industry = c("Industry: type 1",  "Industry: type 10",  "Industry: type 11", "Industry: type 12", "Industry: type 13", "Industry: type 2", 
               "Industry: type 3", "Industry: type 4", "Industry: type 5", "Industry: type 6", "Industry: type 7", "Industry: type 8", "Industry: type 9"), 
  Insurance = c("Insurance"), 
  School = c("Kindergarten", "School", "University"), 
  Legal = c("Legal Services"), 
  Medicine = c("Medicine"),
  Military = c("Military"),
  Mobile = c("Mobile", "Telecom"), 
  Other = c("Other"), 
  P = c("Police"), 
  Postal = c("Postal"), 
  Religion = c("Religion"), 
  Restaurant = c("Restaurant"), 
  Security = c("Security", "Security Ministries"), 
  Services = c("Services"), 
  XNA = c("XNA"),
  Trade = c(
    "Trade: type 1", "Trade: type 2", "Trade: type 3", "Trade: type 4", "Trade: type 5",
    "Trade: type 6", "Trade: type 7"),
  Transport = c(
    "Transport: type 1", "Transport: type 2", "Transport: type 3", "Transport: type 4")
)


#Split data into train and test

table(df$TARGET)

df$TARGET <- factor(df$TARGET)

table(df$TARGET)

predictors <- subset(df, select = -c(TARGET))

#Handling unbalanced data using SMOTE. Using SMOTE we are simulating observations for TARGET = 1 class
#SMOTE requires Predictor(s) and response to be passed separately to it.
smote.obj <- ubSMOTE(X = predictors, Y = df$TARGET, perc.over = 200, k = 3, verbose = FALSE)

beep(sound=3)

#Joining predictors and response into a single df
df <- cbind('TARGET' = smote.obj$Y, smote.obj$X)

set.seed(456)
train_ind <- sample(seq_len(nrow(df)), size = sample.size <- floor(0.7 * nrow(df)))
train <- df[train_ind,]
test <- df[-train_ind,]


library('rpart')
library('rpart.plot')


#Generating tree with minsplit = 50

train.tree <- rpart(train$TARGET~., data=train, control=rpart.control(cp=0, minsplit=50, xval=10), method = "class")
rpart.plot(train.tree, roundint=FALSE)

predict.train.tree <- predict(train.tree, newdata = train, type="class")
confusion.matrix.train <- table(predict.train.tree, train$TARGET)

result <- c(
  "0"=confusion.matrix.train[1]/(confusion.matrix.train[1]+confusion.matrix.train[3]), 
  "1"=confusion.matrix.train[4]/(confusion.matrix.train[2]+confusion.matrix.train[4])
)

result

#Performing Holdout with minsplit = 50


predict.holdout.tree <- predict(train.tree, newdata = test, type="class")
confusion.matrix.holdout <- table(predict.holdout.tree, test$TARGET)

c(
  "0"=confusion.matrix.holdout[1]/(confusion.matrix.holdout[1]+confusion.matrix.holdout[3]), 
  "1"=confusion.matrix.holdout[4]/(confusion.matrix.holdout[2]+confusion.matrix.holdout[4])
)


#############################
#Confusion matrix on prune tree for minsplit =50

printcp(train.tree)
plotcp(train.tree)

#Based on the table and plot, 9.9306e-04 (0.00099) seems to be optimal value to prune the tree.

prune.tree <- prune(train.tree, cp=0.00099)
rpart.plot(prune.tree)

predict.prune.holdout.tree <- predict(prune.tree, newdata = test, type="class")
confusion.matrix.prune.holdout <- table(predict.prune.holdout.tree, test$TARGET)


c(
  "0"=confusion.matrix.prune.holdout[1]/(confusion.matrix.prune.holdout[1]+confusion.matrix.prune.holdout[3]), 
  "1"=confusion.matrix.prune.holdout[4]/(confusion.matrix.prune.holdout[2]+confusion.matrix.prune.holdout[4])
)



library(caret)

#Printing confusion matrix for pruned tree.
confusionMatrix(confusion.matrix.prune.holdout)


tree.summary <- summary(prune.tree)

#Based on the results of the classification tree, Credit Score (EXT_SOURCE_3 seems) to be most important variable in determining the likelihood of an applicant defaulting.


