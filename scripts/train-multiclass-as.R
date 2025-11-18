library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

source("scripts/preprocessing-final.R")

load("data/claims-raw.RData")

claims_clean <- clean_claims(claims_raw) %>%
  filter(!is.na(mclass), text_clean != "")

claims_clean$mclass <- factor(claims_clean$mclass)
L <- length(levels(claims_clean$mclass))

set.seed(222)
split <- initial_split(claims_clean, prop = 0.8, strata = mclass)
train <- training(split)
test  <- testing(split)

train_x <- train$text_clean
train_y <- to_categorical(as.integer(train$mclass) - 1, num_classes = L)

vec <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split = "whitespace",
  output_mode = "tf_idf",
  max_tokens = 12000
)
vec |> adapt(train_x)

input <- layer_input(shape = c(1), dtype = "string")
output <- input |>
  vec() |>
  layer_dense(128, activation = "relu") |>
  layer_dropout(0.3) |>
  layer_dense(L, activation = "softmax")

model <- keras_model(input, output)

model |> compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

model |> fit(
  train_x, train_y,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

save_model_tf(model, "results/multiclass-model")
saveRDS(serialize_model(vec), "results/multiclass-vectorizer.rds")

cat("Finished training multiclass model.\n")
