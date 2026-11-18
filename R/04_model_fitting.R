set.seed(123)

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

if (!require(tidyr)) {
  library(tidyr)
}

if(!require(bayesplot)){
  library(bayesplot)
}

if (!require(priorsense)) {
  install.packages("priorsense") 
  library(priorsense)
}

# pulse <- readRDS(file.path(root, "data", "processed", "pulse.rds"))

# dummy transforms AND ordering
# VARIABLE SELECTION (LECTURE 11)
model_vars <- c("low3", "age", "age_num", "econ", "econ5", "edu", "region")
# see NAs!
dat <- pulse %>%
  drop_na(low3) %>%
  mutate(
    edu   = relevel(factor(edu,   ordered = FALSE), ref = "basic"),
    econ5 = relevel(factor(econ5, ordered = FALSE), ref = "cant_say"),
    econ  = factor(econ, ordered = TRUE),
    age   = factor(age,  ordered = FALSE),
    region = factor(region)
  )

# ALSO -> two-way interactions between the covariates (interaction effect)
# add gender / language as covariates and compare with LOO!
# ALSO -> ordinal model on the raw 1-10 scale -> family = cumulative("logit") with
# mo(econ); then derive P(y <= 3) from the cumulative probabilities -> WHY?
# https://stats.stackexchange.com/questions/280270/logistic-regression-vs-ordinal-regression-using-cumulative-logits

# regular model -> use regular logistic regression (0/1)
fit_opts <- function(formula, prior) {
  brm(formula = formula, data = dat, family = bernoulli("logit"),
      prior = prior, backend = "cmdstanr",
      chains = 4, iter = 2000, cores = 4)
}

# simple initial, weakly informative priors -> effects should 
# not change the log odds too much!
pr_base <- c(prior(normal(0, 1), class = "b"),
             prior(normal(0, 2), class = "Intercept"))

# prior limit from 0 -> exponentially smaller
# https://avehtari.github.io/BDA_course_Aalto/project_gsu.html#model-requirements
pr_re   <- c(pr_base, prior(exponential(1), class = "sd"))
# 8. Distance parameters of monotonic effects in "https://paulbuerkner.com/brms/reference/set_prior.html"
# for mo() -> only valid is the dirichlet(1)
pr_mo   <- c(pr_base, prior(dirichlet(1), class = "simo", coef = "moecon1"))
pr_mo_re<- c(pr_mo,   prior(exponential(1), class = "sd"))
# 4. Smoothing Splines -> variability within, use normal for weakly informative
pr_s    <- c(pr_base, prior(normal(0, 1), class = "sds"))
pr_full <- c(pr_base, prior(dirichlet(1), class = "simo", coef = "moecon1"),
             prior(normal(0, 1), class = "sds"),
             prior(exponential(1), class = "sd"))

m1 <- fit_opts(low3 ~ age + econ5 + edu, pr_base)
# VARYING INTERCEPT -> motivate
m2 <- fit_opts(low3 ~ age + econ5 + edu + (1 | region), pr_re)
m3 <- fit_opts(low3 ~ age + mo(econ) + edu, pr_mo)
m4 <- fit_opts(low3 ~ s(age_num) + econ5 + edu,   pr_s)
m5 <- fit_opts(low3 ~ s(age_num) + mo(econ) + edu + (1 | region),
               pr_full)
# motivate this -> varying econ slope and intercept?
m6 <- fit_opts(low3 ~ age + mo(econ) + edu + (1 + mo(econ) | region),
               pr_mo_re)

# did the chains convergence, do they need to be tuned?
# non-centered param -> for HMC convergence if necessary?
# HMC diagnostics?
# SIMPLE SUMMARY -> NO ISSUES / WHAT WERE THE PROBLEMATIC PARTS?
summary(m1)

# marginal posterior for coefficients reported?
# report + interpret marginal coefficient posteriors (log-odds / odds ratios)!
# CONDITIONAL EFFECTS -> what proportion is k≤3
plot(conditional_effects(m1), ask = FALSE)
# coefficient / odds-ratio interval plot -> for coefficient comparison
mcmc_plot(m1, variable = "^b_", regex = TRUE, type = "intervals", prob_outer = 0.95)

# fits <- list(M1 = m1, M2 = m2, M3 = m3, M4 = m4, M5 = m5, M6 = m6)
# summary(m2)
# summary(m3)
# summary(m4)
# summary(m5)
# summary(m6)

# IN THE REPORT -> draw from the LOO predictive distribution
# reported PPC is the calibration (PAV / CEP)
# https://journals.aau.dk/index.php/jovi/article/view/11478/9117
# https://mc-stan.org/bayesplot/articles/articles-online-only/ppc-calibration.html
ll_m1    <- log_lik(m1)
chain_id <- rep(seq_len(nchains(m1)), each = nrow(ll_m1) / nchains(m1))
r_eff_m1 <- loo::relative_eff(exp(ll_m1), chain_id = chain_id)
psis_m1  <- loo::psis(-ll_m1, r_eff = r_eff_m1)
w_m1     <- weights(psis_m1, log = FALSE)
yrep_m1  <- posterior_predict(m1)
S_m1     <- nrow(yrep_m1)
# resample each column by its PSIS weights -> draws from the LOO predictive
m1_loo_draws <- sapply(seq_len(ncol(yrep_m1)), function(i) {
  yrep_m1[sample.int(S_m1, S_m1, replace = TRUE, prob = w_m1[, i]), i]
})

# REPORTED CALIBRATION (PPC)
# article figure 6 comparison for interpretation (calibration curve, pred. mass, interval bands)
# AI -> "consider grouped calibration (ppc_calibration faceted by region / econ) since the 
# research question is about low trust within subgroups"
ppc_calibration(y = m1$data$low3, yrep = m1_loo_draws,
                interval = "confidence", prob = 0.95)

# PSIS_LOO -> elpd score and uncertainty
# "Report and interpret the comparison results, including uncertainty in the estimates"
m1 <- add_criterion(m1, c("loo"))
loo(m1)
#          Estimate   SE
# elpd_loo   -980.0 27.3
# p_loo        17.8  0.9
# REPORT ALSO -> MCSE of elpd_loo is 0.1.
# MCSE and ESS estimates assume MCMC draws (r_eff in [0.3, 1.9]).
# All Pareto k estimates are good (k < 0.7).

# PRIOR PREDICTIVE CHECKS -> group by underlying variable?
powerscale_sensitivity(m1)
powerscale_plot_dens(m1)
powerscale_plot_ecdf(m1)

# (bayes_R2(m1) -> could include)

# classification scores (accuracy, F1, Brier, etc?) -> "Where relevant to the research question,
# also assess predictive performance using practical measures (e.g., classification accuracy) and
# discuss their practical usefulness"
# AI -> compute these from the LOO predictive (reuse m1_loo_draws) -> out-of-sample