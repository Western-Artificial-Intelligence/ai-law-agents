#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
  library(readr)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(input = NULL, out = NULL, ci = "profile", level = 0.95)
  for (a in args) {
    kv <- strsplit(a, "=", fixed = TRUE)[[1]]
    if (length(kv) == 2) {
      key <- sub("^--", "", kv[1])
      val <- kv[2]
      if (key == "input") out$input <- val
      else if (key == "out") out$out <- val
      else if (key == "ci") out$ci <- val
      else if (key == "level") out$level <- as.numeric(val)
    }
  }
  if (is.null(out$input)) stop("--input=path/to/outcomes.csv is required", call. = FALSE)
  out
}

args <- parse_args()
df <- readr::read_csv(args$input, show_col_types = FALSE)
df$pair_id <- as.factor(df$pair_id)

fit <- suppressMessages(
  lme4::glmer(verdict_bin ~ cue_treatment + (1 | pair_id),
              data = df, family = binomial(link = "logit"),
              control = glmerControl(optimizer = "bobyqa"))
)

beta <- as.numeric(lme4::fixef(fit)["cue_treatment"])
se_wald <- suppressWarnings(sqrt(diag(vcov(fit)))["cue_treatment"]) 
ci_logit <- c(NA_real_, NA_real_)
method_used <- args$ci

if (!is.na(se_wald) && (args$ci == "wald")) {
  z <- qnorm(1 - (1 - args$level)/2)
  ci_logit <- c(beta - z * se_wald, beta + z * se_wald)
} else {
  method_used <- "profile"
  suppressWarnings({
    ci <- tryCatch(
      confint(fit, parm = "cue_treatment", method = "profile", level = args$level),
      error = function(e) NULL
    )
  })
  if (!is.null(ci) && nrow(ci) == 1) {
    ci_logit <- as.numeric(ci[1, ])
  } else if (!is.na(se_wald)) {
    method_used <- "wald"
    z <- qnorm(1 - (1 - args$level)/2)
    ci_logit <- c(beta - z * se_wald, beta + z * se_wald)
  }
}

res <- list(
  model = "glmer_logit",
  n = nrow(df),
  method_ci = method_used,
  level = args$level,
  beta = beta,
  se_wald = se_wald,
  or = exp(beta),
  ci_logit = ci_logit,
  ci_or = exp(ci_logit)
)

if (!is.null(args$out)) {
  jsonlite::write_json(res, args$out, auto_unbox = TRUE, digits = NA)
} else {
  cat(jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA))
}
