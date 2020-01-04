library(Hmisc)
library(caret)
library(rms)
library(randomForest)

training_data <- read.csv("/Users/admin/Downloads/projects/kik_challenge/data/training_file.csv")
challenge_data <- read.csv("/Users/admin/Downloads/projects/kik_challenge/data/challenge.csv")

head(training_data, n = 5)
nrow(training_data)

nrow(challenge_data)
head(challenge_data, n = 5)

complete_training_data <- na.omit(training_data)
complete_training_data$good <- NULL

# hc <- findCorrelation(abs(cor(complete_data, method = "spearman")), cutoff = 0.7)
# hc <- sort(hc)
# reduced_data <- complete_data[,-c(hc)]
# print(reduced_data)


# First, we use k-means clustering with 2 centers on challenge data
complete_challenge_data <- na.omit(challenge_data)
challenge_cluster <- kmeans(complete_challenge_data, 2, nstart = 20)
# We use k-means to create labels that are true/false

new_challenge_data <- data.frame(complete_challenge_data, good = challenge_cluster$cluster)
ncol(new_challenge_data)
nrow(new_challenge_data)
table(new_challenge_data$good)
new_challenge_data <- as.factor(new_challenge_data$good)

# We build a random forest model for prediction using the training data
complete_training_data <- na.omit(training_data)
complete_training_data$good <- as.factor(as.numeric(complete_training_data$good))

rf_model <- randomForest(complete_training_data$good~., data = complete_training_data, ntree = 100)

downsample_training_data <- downSample(complete_training_data, '1')

table(predict(rf_model, newdata = complete_challenge_data, type = "class"))
pred_result <- predict(rf_model, newdata = complete_challenge_data, type = "response")

length(predict(rf_model, newdata = complete_challenge_data, type = "response"))

complete_challenge_data <- data.frame(complete_challenge_data, good = pred_result)
nrow(complete_challenge_data)
# Then, we use the random forest model on the new challenge data with labels

levels(complete_training_data$good) <- c("bad", "good")

names(complete_training_data) <- make.names(names(complete_training_data))
table(complete_training_data$good)
nmin <- sum(complete_training_data$good == "bad")
ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = 
                       twoClassSummary)
set.seed(40)
rfDownsampled <- train(good ~ ., 
                       data = complete_training_data,
                       method = "rf",
                       ntree = 100,
                       tuneLength = 5,
                       metric = "ROC",
                       trControl = ctrl,
                       strata = complete_training_data$good,
                       sampsize = rep(nmin, 2))
plot(rfDownsampled)
table(predict(rfDownsampled, newdata = complete_challenge_data, type = "raw"))

rfDownsampled$results





