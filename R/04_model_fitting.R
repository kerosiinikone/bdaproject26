# Testing several models for (TBD) coefficients

# Outcome -> trust (q11k) as binary target -> k <= 3
# (1) -> few covariates for the logit(k <= 3 / k > 3), simple, weakly informative normal(0, 1) priors
# eda_02 -> economic situation, age group and education have most variance in the proportion of low trust (FIRST MODEL)
# eda_03 -> grouped by region

if(!require(cmdstanr)){
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
  library(cmdstanr)
}

if (!require(brms)) {
  library(brms)
}

if (!require(dplyr)) {
  library(dplyr)
}

if(!require(ggplot2)){
  library(ggplot2)
}

if(!require(bayesplot)){
  library(bayesplot)
}

# logit transformation for q11k <= 3
pulse <- pulse %>%
  mutate(q11k_binary = ifelse(q11k < 3, 1, 0))

# dummy variables -> changes in the log-odds
pulse <- pulse %>%
  mutate(
    age = factor(age),
    econ5 = relevel(factor(econ5), ref = "cant_say"),
    edu = relevel(factor(edu), ref = "basic")
)

# normal for coefficients (weakly inf.) -> bind
# wider normal for intercept (weakly inf.)
priors_normal <- c(
  prior(normal(0, 2), class="Intercept"),
  prior(normal(0, 1), class = "b")
)

q11k_k3_fit1 <- brm(
  formula = q11k_binary ~ age + econ5 + edu,
  family = bernoulli(link = "logit"),
  data = pulse,
  prior = priors_normal,
  backend = "cmdstanr",
  chains = 4,
  cores = 4
)

summary(q11k_k3_fit1)

# PAV-adjusted calibration (according to lectures, https://journals.aau.dk/index.php/jovi/article/view/11478/9117)
# regular accuracy, precision, AUC metrics?

