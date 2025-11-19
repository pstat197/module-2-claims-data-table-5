library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)
source("scripts/preprocessing-final.R")


load("data/claims-raw.RData")

claims_clean <- clean_claims(claims_raw) %>%
  filter(!is.na(bclass), text_clean != "")

claims_clean$bclass <- factor(claims_clean$bclass)

set.seed(111)
split <- initial_split(claims_clean, prop = 0.8, strata = bclass)
train <- training(split)
test  <- testing(split)

train_x <- train$text_clean
train_y <- ifelse(train$bclass == levels(train$bclass)[2], 1L, 0L)

vec <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split = "whitespace",
  output_mode = "tf_idf",
  max_tokens = 10000
)
vec |> adapt(train_x)

input <- layer_input(shape = c(1), dtype = "string")
output <- input |> 
  vec() |>
  layer_dense(64, activation = "relu") |>
  layer_dropout(0.2) |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

model |> compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "binary_accuracy"
)

model |> fit(
  train_x, train_y,
  validation_split = 0.2,
  epochs = 8,
  batch_size = 32
)
save_model_tf(model, "results/binary-model")
saveRDS(serialize_model(vec), "results/binary-vectorizer.rds")

cat("Finished training binary model.\n")
