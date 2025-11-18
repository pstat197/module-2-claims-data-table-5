library(tidyverse)
library(keras3)
source("scripts/preprocessing-final.R")

load("data/claims-test.RData")
load("data/claims-raw.RData")

clean_test <- clean_claims(claims_test)

bin_model <- load_model_tf("results/binary-model")
bin_vec   <- unserialize_model(readRDS("results/binary-vectorizer.rds"))

multi_model <- load_model_tf("results/multiclass-model")
multi_vec   <- unserialize_model(readRDS("results/multiclass-vectorizer.rds"))

# Binary predictions
bin_prob <- predict(bin_model, clean_test$text_clean)
bin_levels <- levels(claim
                     