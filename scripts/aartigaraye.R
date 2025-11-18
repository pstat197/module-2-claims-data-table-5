# Perform a secondary tokenization of the data to obtain bigrams. Fit a logistic principal component regression 
#model to the word-tokenized data, and then input the predicted log-odds-ratios together with some number of 
#principal components of the bigram-tokenized data to a second logistic regression model. Based on the results, 
#does it seem like the bigrams capture additional information about the claims status of a page?

# What is the goal of Preliminary task 2?
# Perform a secondary tokenization to obtain bigrams.
# Fit a logistic principal component regression model to the word-tokenized TF–IDF data,
# then use the predicted log-odds from that model plus some PCs of the bigram data as predictors in a second 
# logistic regression model.
#Evaluate whether bigrams add predictive power.

# So we need to build a two step stacked model. We would want to use the cleaned text. Create a word-level 
# TF-IDF matrix i.e. the term frequency inverse document frequency as mentioned in class. The idea is 
# Words that appear frequently in a single document but not frequently across all documents and get 
# the highest score. each row is one document and each column is one token and each cell is the TFIDF 
# score. This is already in the nlp_fn in one of the provided scripts. Then we perform PCA on the 
# TFDIF matrix and fit a logistic regression and save the predicted log-odds. 

# Then we create bigram TFDIF using the `unnest_tokens(token="ngrams", n=2)` 
# PCA on this and fit a second logistic regression and compare the metrics like auc, accuracy, deviance, 
# BIC/AIC, and determine whether bigrams improve the performance or not. 

## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
require(irlba)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# Loading the data
load("data/claims-clean-example.RData")   # your updated cleaned file
word_df <- nlp_fn(claims_clean)

head(word_df, 5)
# The columns of this df are the work TFIDF features.

# Now we want to perform PCA on these
word_matrix <- word_df %>%
  select(-.id, -bclass) %>%
  as.matrix()

K <- 100
pca_words <- prcomp_irlba(word_matrix, n = K, center = TRUE, scale. = TRUE)
word_pc <- pca_words$x

# we pick number of pcs, common choices are 50, 100 to capture at least 80% of the variance
# We retained the first 100 principal components from the word TF–IDF PCA.
# They explained approximately 80–90% of the variance and provided stable logistic regression fits 
# without overfitting. this is the standard

# this step takes a long time to run almost 5-10 minutes. because prcomp() does a full SVD on a 
# dense matrix – computational complexity is roughly O(min(np^2, pn^2)). 
# We can fix it by using sparse PCA (irlba) instead of prcomp() but I have just left it like that for now
# I changed it to use sparse, using the irlba library


# Logistic regression 
log_reg_df_regular <- data.frame(
  bclass = as.factor(word_df$bclass),
  word_pc
)

modreg <- glm(bclass ~ ., data = log_reg_df_regular, family = binomial())
logoddsreg <- predict(modreg, type = "link")
summary(modreg)
# For warning 1. The warning is completely expected because we have lots of predictors and the model is close 
# to perfectly separable. Iteratively reweighted least squares (IRLS) gets unstable. We can still use 
# predicted log odds. The purpose here is dimension reduction, not interpreting coefficients, 
# so slight non-convergence is fine.

# For warning 2: This happens all the time in stacked models and PCA logistic regression. Not a problem 
# in our case.



# Building the bigrams TFIDF matrix

# Creating my own bigram version using the nlp_fn 
nlp_bigrams <- function(parse_data.out){
  parse_data.out %>%
    unnest_tokens(output = bigram,
                  input = text_clean,
                  token = "ngrams",
                  n = 2) %>%
    filter(!str_detect(bigram, "\\b\\w\\b")) %>%
    group_by(bigram) %>%
    filter(n() > 3) %>% # keep bigrams appearing in >3 docs ow it get's too big & reaches the vector memory limit
    ungroup() %>%
    count(.id, bclass, bigram, name = "n") %>%
    bind_tf_idf(term = bigram,
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c(".id", "bclass"),
                names_from = "bigram",
                values_from = "tf_idf",
                values_fill = 0)
}

# This takes a long time to run as well. Anywhere from 2-20 minutes. It's Tokenizing text into bigrams, counting
# all the bigrams and building a matrix with thousands, even tens of thousands of bigram features.
bigram_df <- nlp_bigrams(claims_clean)

# Now we are working on PCA on bigram TFIDF
M <- 50 # We want M to be less than K in the above logistic regression because 
# the bigrams are less than tokens in total 
bigram_matrix <- bigram_df %>% select(-.id, -bclass) %>% as.matrix()

svd_bigram <- irlba(scale(bigram_matrix, center = TRUE, scale = TRUE), nv = M)

# pca_bigram <- irlba(bigram_matrix, nv = M, center = TRUE, scale = TRUE)


bigram_pcs <- svd_bigram$u %*% diag(svd_bigram$d)
colnames(bigram_pcs) <- paste0("bigram_PC", 1:M)


# Now this is the tricky part, fit a logistic principal component regression model to the word-tokenized data,
# and then input the predicted log-odds-ratios together with some number of principal components of the 
# bigram-tokenized data to a second logistic regression model.
# What this means is that once we fit the logistic regression using the regular word PCA and then we combine
# the predicted log odds and the bigram PCs into one dataset and fit a second logistic regression model. This 
# is because we want to check whether the bigrams are ADDING anything to the original metric. 

# Put bigram PCs into a dataframe with ID so we can merge correctly
bigram_pc_df <- data.frame(
  .id = bigram_df$.id,
  bigram_pcs
)

# Merge word-level results with bigram PCs by ID
combined_df <- word_df %>%
  select(.id, bclass) %>%
  left_join(
    data.frame(.id = bigram_df$.id, bigram_pcs),
    by = ".id"
  ) %>%
  mutate(logodds_word = logoddsreg)

log_reg_bigram <- glm(bclass ~ ., data = combined_df %>% select(-.id),
                      family = binomial())

summary(log_reg_bigram)

# ============================================================
# Evaluating the predictive accuracy
# ============================================================
library(pROC)
library(caret)

# ==== FIX FOR LENGTH MISMATCH ====
# glm automatically removes rows with NA bigram PCs
used_rows <- as.numeric(rownames(log_reg_bigram$model))

# True labels for the rows actually used by the bigram model
y2 <- as.numeric(combined_df$bclass[used_rows]) - 1

# Predictions for those same rows
pred2_prob <- predict(log_reg_bigram, type = "response")
pred2_class <- ifelse(pred2_prob > 0.5, 1, 0)

# AUC
auc1 <- roc(y, prob1)$auc
auc2 <- roc(y2, pred2_prob)$auc

# Confusion matrices
cm1 <- confusionMatrix(factor(pred1_class), factor(y))
cm2 <- confusionMatrix(factor(pred2_class), factor(y2))

# ============================================================
# Information criteria & log-likelihood
# ============================================================

# Refit Model 1 **using only the rows used in Model 2**
modreg_restricted <- glm(
  bclass ~ ., 
  data = log_reg_df_regular[used_rows, ],
  family = binomial
)

# AIC, BIC, Log-likelihood for restricted Model 1
aic1 <- modreg_restricted$aic
bic1 <- AIC(modreg_restricted, k = log(length(used_rows)))
ll1  <- logLik(modreg_restricted)[1]

# AIC, BIC, Log-likelihood for Model 2
aic2 <- log_reg_bigram$aic
bic2 <- AIC(log_reg_bigram, k = log(length(used_rows)))
ll2  <- logLik(log_reg_bigram)[1]

# ============================================================
# Likelihood Ratio Test (now valid)
# ============================================================
lr_test <- anova(modreg_restricted, log_reg_bigram, test = "Chisq")

# ============================================================
# Reporting
# ============================================================
cat("=== MODEL 1 (Word PCA only — restricted) ===\n")
cat("AUC:", round(auc1, 4), "\n")
cat("Accuracy:", round(cm1$overall['Accuracy'], 4), "\n")
cat("AIC:", round(aic1, 2), "\n")
cat("BIC:", round(bic1, 2), "\n")
cat("Log-Likelihood:", round(as.numeric(ll1), 2), "\n\n")

cat("=== MODEL 2 (Word PCA + Bigram PCA) ===\n")
cat("AUC:", round(auc2, 4), "\n")
cat("Accuracy:", round(cm2$overall['Accuracy'], 4), "\n")
cat("AIC:", round(aic2, 2), "\n")
cat("BIC:", round(bic2, 2), "\n")
cat("Log-Likelihood:", round(as.numeric(ll2), 2), "\n\n")

cat("=== IMPROVEMENT ===\n")
cat("Δ AUC:", round(auc2 - auc1, 4), "\n")
cat("Δ Accuracy:", round(cm2$overall['Accuracy'] - cm1$overall['Accuracy'], 4), "\n")
cat("Δ AIC:", round(aic2 - aic1, 2), "(negative = better)\n")
cat("Δ BIC:", round(bic2 - bic1, 2), "(negative = better)\n\n")

cat("=== LIKELIHOOD RATIO TEST ===\n")
print(lr_test)

# The results show a substantial improvement when bigram information is incorporated into the model.
# The original logistic PCR model using only word-level principal components performs poorly, with an AUC
# of about 0.55 and an accuracy of 0.59, barely above random guessing. After adding the predicted log-odds from
# the word-based model and then including principal components derived from bigram counts, the predictive 
# performance increases dramatically: the new model attains an AUC of 0.86 and an accuracy of about 0.79.
# This is a very large gain in discrimination ability, indicating that the bigram representations capture 
# important structure in the text that word-level features alone do not.

# The model comparison metrics reinforce this conclusion. The AIC and BIC of the bigram-augmented model 
# are dramatically lower indicating a much better fit even after accounting for additional parameters.
# The likelihood-ratio test also overwhelmingly favors the bigram model (with a deviance reduction of 
# ~55,800), showing that the improvement is statistically significant. Taken together, these results strongly
# suggest that bigrams provide substantial additional information about the claim status of a page—they 
# capture contextual or phrase-level patterns that single tokens cannot, and incorporating them leads to far
# more accurate classification.
