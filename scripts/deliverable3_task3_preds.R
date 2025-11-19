# Deliverable 3 for Preliminary Task 3 (Group 5)

library(tidyverse)


prob_task3 <- predict(model, test_text) |> as.vector()

binary_pred <- ifelse(prob_task3 > 0.5, lev[2], lev[1]) |>
  factor(levels = lev)

pred_df <- tibble(
  .id = test_set$.id,
  bclass.pred = binary_pred,
  mclass.pred = NA_character_
)

save(pred_df, file = "results/preds-group5_task3.RData")
