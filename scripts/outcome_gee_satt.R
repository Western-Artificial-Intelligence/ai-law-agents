#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(geepack)
  library(clubSandwich)
  library(jsonlite)
  library(readr)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(input = NULL, out = NULL, level = 0.95)
  for (a in args) {
    kv <- strsplit(a, "=", fixed = TRUE)[[1]]
    if (length(kv) == 2) {
      key <- sub("^--", "", kv[1])
      val <- kv[2]
      if (key == "input") out$input <- val
      else if (key == "out") out$out <- val
      else if (key == "level") out$level <- as.numeric(val)
    }
  }
  if (is.null(out$input)) stop("--input=path/to/outcomes.csv is required", call. = FALSE)
  out
}

args <- parse_args()
df <- readr::read_csv(args$input, show_col_types = FALSE)

df$pair_id <- as.factor(df$pair_id)

fit <- suppressMessages(geepack::geeglm(
  verdict_bin ~ cue_treatment,
  id = pair_id, data = df, family = binomial(link = "logit"), corstr = "exchangeable"
))

V <- clubSandwich::vcovCR(fit, cluster = df$pair_id, type = "CR2")
ct <- clubSandwich::coef_test(fit, vcov = V, test = "Satterthwaite")
row <- ct[rownames(ct) == "cue_treatment", , drop = FALSE]

beta <- as.numeric(row$beta)
se <- as.numeric(row$SE)
df_satt <- as.numeric(row$df_Satt)
tcrit <- suppressWarnings( qt(1 - (1 - args$level)/2, df = df_satt) )
ci_logit <- c(beta - tcrit * se, beta + tcrit * se)
res <- list(
  model = "gee_logit_cr2_satt",
  n = nrow(df),
  level = args$level,
  beta = beta,
  se_cr2 = se,
  df_satt = df_satt,
  or = exp(beta),
  ci_logit = ci_logit,
  ci_or = exp(ci_logit)
)
if (!is.null(args$out)) {
  jsonlite::write_json(res, args$out, auto_unbox = TRUE, digits = NA)
} else {
  cat(jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA))
}
