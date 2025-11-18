library(tidyverse)
library(rvest)
library(qdapRegex)
library(stringr)

extract_text <- function(html) {
  doc <- tryCatch(read_html(html), error = function(e) NA)
  if (all(is.na(doc))) return("")
  
  texts <- doc %>%
    html_elements("h1, h2, h3, h4, h5, h6, p") %>%
    html_text2()
  
  txt <- paste(texts, collapse = " ")
  
  txt %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_replace_all("[[:digit:]]", " ") %>%
    str_squish() %>%
    tolower()
}

clean_claims <- function(df) {
  df %>%
    mutate(
      text_clean = map_chr(html, extract_text),
      text_clean = str_squish(text_clean)
    )
}
