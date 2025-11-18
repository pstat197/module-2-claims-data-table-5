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


bin_prob <- predict(bin_model, clean_test$text_clean)

bin_levels <- levels(factor(claims_raw$bclass))

bin_pred <- ifelse(bin_prob > 0.5, bin_levels[2], bin_levels[1])
bin_pred <- factor(bin_pred, levels = bin_levels)

multi_prob <- predict(multi_model, clean_test$text_clean)

multi_levels <- levels(factor(claims_raw$mclass))

multi_pred <- apply(multi_prob, 1, function(row) {
  multi_levels[which.max(row)]
})
multi_pred <- factor(multi_pred, levels = multi_levels)


pred_df <- tibble(
  .id = clean_test$.id,
  bclass.pred = bin_pred,
  mclass.pred = multi_pred
)

save(pred_df, file = "results/preds-group5.RData")

cat("Saved predictions to results/preds-group5.RData\n")
