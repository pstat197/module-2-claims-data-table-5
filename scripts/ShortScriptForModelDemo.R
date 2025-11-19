library(tidyverse)
library(tidytext)
library(textstem)
library(irlba)

# Task 1 Models and Predictions on the Test data

# Task 2 Models and Predictions

load("results/model_word_pca_task2.RData")   # pca_words + modreg
load("results/model_bigram_pca_task2.RData") # svd_bigram
load("results/model_stack_task2.RData")  # log_reg_bigram

# Generate predictions on the data you already used
# (just a demonstration)
pred_word <- predict(modreg, type = "response")
pred_stacked <- predict(log_reg_bigram, type = "response")

# Print first 6 predictions
print(head(pred_word))
print(head(pred_stacked))
# Each value is the model’s predicted probability of the positive class for the observation. 
# The predicted stack probabilities are for te model with bigrams. Each row of your data 
# corresponds to one document / webpage. The response variable is `bclass` s a binary label for the document, e.g., 
# whether the page is a “claim” or not. The probabilities are that the document belongs to class 1 or not. 
# In other words, the probability reflects the model’s confidence that the document belongs to class 1.

# Task 3 Models and Predictions

# Primary Task Model and Prediction