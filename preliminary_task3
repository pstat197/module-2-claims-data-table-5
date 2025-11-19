## PSTAT 197A – Module 2, Preliminary Task 3

library(tidyverse)
library(tidytext)
library(rsample)
library(tensorflow)
library(keras3) 

set.seed(197)


## load in our data


load("data/claims-clean-example.RData")   # creates claims_clean

clean <- claims_clean %>%
  filter(
    !is.na(bclass),
    !is.na(text_clean),
    str_trim(text_clean) != ""
  ) %>%
  mutate(
    bclass = factor(bclass),
    text_clean = str_trim(text_clean)
  )

cat("Rows in clean:", nrow(clean), "\n")
cat("Classes in bclass:", paste(levels(clean$bclass), collapse = ", "), "\n")


## train/test split

partitions <- initial_split(clean, prop = 0.8, strata = bclass)

train_set <- training(partitions)
test_set  <- testing(partitions)

cat("Training rows:", nrow(train_set), "\n")
cat("Test rows    :", nrow(test_set), "\n")

train_text <- train_set |> pull(text_clean)
test_text  <- test_set  |> pull(text_clean)

Y_train_factor <- train_set |> pull(bclass)
Y_test_factor  <- test_set  |> pull(bclass)

# factor levels
Y_train_factor <- factor(Y_train_factor)
Y_test_factor  <- factor(Y_test_factor, levels = levels(Y_train_factor))

lev <- levels(Y_train_factor)
positive_class <- lev[2]

Y_train <- ifelse(Y_train_factor == positive_class, 1L, 0L)

preprocess_layer <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split       = "whitespace",
  ngrams      = NULL,
  max_tokens  = 10000,          
  output_mode = "tf_idf"
)

# adapt on training text
preprocess_layer %>% adapt(train_text)


input <- layer_input(
  shape = c(1),
  dtype = "string",
  name  = "text_input"
)


output <- input |>
  preprocess_layer() |>
  layer_dropout(rate = 0.1) |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = input, outputs = output)

summary(model)

model |>
  compile(
    loss      = "binary_crossentropy",
    optimizer = "adam",
    metrics   = "binary_accuracy"
  )

history <- model |>
  fit(
    x = train_text,        
    y = Y_train,
    epochs = 10,           # can tune
    batch_size = 32,
    validation_split = 0.2,
    verbose = 1
  )

plot(history)   # train vs val accuracy

prob <- predict(model, test_text) |> as.vector()

pred <- ifelse(prob > 0.5, lev[2], lev[1]) |>
  factor(levels = lev)

nn3_acc <- mean(pred == Y_test_factor)

cat("\nNeural network TF–IDF test accuracy (nn3_acc):",
    round(nn3_acc, 4), "\n")

## Here, we are comparing to our logistic PCA from preliminary task 1

acc_par <- 0.54   # preliminary task 1 accuracy

results <- tibble(
  model    = c("logistic_PCA", "NN_tf-idf"),
  accuracy = c(acc_par,       nn3_acc)
)

cat("\nAccuracy comparison:\n")
print(results)

## The TF-IDF neural network achieved about 0.785 accuracy, which is substantially higher than the PCA logistic regression accuracy of 0.54. This suggests that the neural network can capture more useful signal from the text than PCA, which compresses the features too aggressively and loses predictive information.

keras3::save_model(model, "results/nn_task3_binary.keras")

cat("✅ Saved NN model to results/nn_task3_binary.keras\n")
