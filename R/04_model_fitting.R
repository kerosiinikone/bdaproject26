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

if (!require(posterior)) {
  library(posterior)
}

if (!require(priorsense)) {
  install.packages("priorsense") 
  library(priorsense)
  
}
library(loo)


pulse <- readRDS(file.path("data", "processed", "pulse.rds"))

# dummy transforms AND ordering
# VARIABLE SELECTION (LECTURE 11)
# AGE AND ANSWER LANGUAGE -> add!
model_vars <- c("low3", "gender", "age", "age_num", "econ", "edu", "region", 
                "language", "living", "region")
# see NAs!
dat <- pulse %>%
    drop_na(all_of(model_vars)) %>%
  mutate(
    gender   = relevel(factor(gender), ref = "woman"),
    age      = relevel(factor(age, ordered = FALSE), ref = "60-64"),
    econ_f   = relevel(factor(econ, ordered = FALSE), ref = "middle"),  # dummies, M1/M2
    econ     = factor(econ, ordered = TRUE),                            # ordered, mo() in M3
    edu      = relevel(factor(edu), ref = "upper_secondary"),
    language = relevel(factor(language), ref = "finnish"),
    living   = relevel(factor(living), ref = "family"),
    region   = factor(region)
  )


nrow(dat)

# ALSO -> two-way interactions between the covariates (interaction effect)
# add gender / language as covariates and compare with LOO!
# ALSO -> ordinal model on the raw 1-10 scale -> family = cumulative("logit") with
# mo(econ); then derive P(y <= 3) from the cumulative probabilities -> WHY?
# https://stats.stackexchange.com/questions/280270/logistic-regression-vs-ordinal-regression-using-cumulative-logits

# regular model -> use regular logistic regression (0/1)
fit_opts <- function(formula, prior, name, ...) {
  brm(formula = formula, data = dat, family = bernoulli("logit"),
      prior = prior, backend = "cmdstanr",
      chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 123,
      file = here::here("output", "fits", name), file_refit = "on_change", ...)
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

# all relevant covariates kept -> not preselected without reason
# m1 <- fit_opts(low3 ~ age + econ5 + edu + language + living, pr_base)
# Modified m1 now with all covariates
m1 <- fit_opts(low3 ~ gender + age + econ_f + edu + language + living,
               pr_base, "m1")
# VARYING INTERCEPT -> motivate
# m2: + varying intercept per region (hierarchical, partial pooling)
m2 <- fit_opts(low3 ~ gender + age + econ_f + edu + language + living +
                 (1 | region), pr_re, "m2",
                 control = list(adapt_delta = 0.95))   # 1 divergence with default 0.8
# m3: m2 with smooth age and monotonic income instead of dummies
m3 <- fit_opts(low3 ~ gender + s(age_num) + mo(econ) + edu + language + living +
                 (1 | region), pr_full, "m3")
# m3 <- fit_opts(low3 ~ age + mo(econ) + edu, pr_mo)
# m4 <- fit_opts(low3 ~ s(age_num) + econ5 + edu,   pr_s)
# m5 <- fit_opts(low3 ~ s(age_num) + mo(econ) + edu + (1 | region),
#                pr_full)
# # motivate this -> varying econ slope and intercept?
# m6 <- fit_opts(low3 ~ age + mo(econ) + edu + (1 + mo(econ) | region),
#                pr_mo_re)

#Convergence: R-hat, ESS, divergences for every model --------------------
conv_check <- function(fit) {
  s  <- summarise_draws(as_draws_df(fit), default_convergence_measures())
  np <- nuts_params(fit)
  data.frame(max_rhat     = max(s$rhat, na.rm = TRUE),
             min_ess_bulk = min(s$ess_bulk, na.rm = TRUE),
             min_ess_tail = min(s$ess_tail, na.rm = TRUE),
             divergences  = sum(np$Value[np$Parameter == "divergent__"]))
}
conv <- rbind(m1 = conv_check(m1), m2 = conv_check(m2), m3 = conv_check(m3))
conv
write.csv(conv, here::here("output", "convergence.csv"))



# did the chains convergence, do they need to be tuned?
# non-centered param -> for HMC convergence if necessary?
# HMC diagnostics?
# SIMPLE SUMMARY -> NO ISSUES / WHAT WERE THE PROBLEMATIC PARTS?
# COEFFICIENTS, PLOTS as appendices
# report artifacts -> output/m1/ (read back into report/report.qmd)
#
# Marginal posteriors
fits <- list(m1 = m1, m2 = m2, m3 = m3)
for (name in names(fits)) {
  fit     <- fits[[name]]
  res_dir <- here::here("output", name)
  dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

  # predicted probability of low trust by each covariate (others at reference)
  p_cond <- plot(conditional_effects(fit), ask = FALSE, plot = FALSE)
  ggplot2::ggsave(file.path(res_dir, "coef_conditional.png"),
                  patchwork::wrap_plots(p_cond), width = 9, height = 6, dpi = 150)

  # coefficients on the log-odds scale; b_ = dummies, bsp_ = monotonic, bs_ = spline
  p_int <- mcmc_plot(fit, variable = "^(b_|bsp_|bs_)", regex = TRUE,
                     type = "intervals", prob_outer = 0.95)
  ggplot2::ggsave(file.path(res_dir, "coef_intervals.png"), p_int,
                  width = 7, height = 6, dpi = 150)

    # region effects after partial pooling (only models with (1 | region))
  if (any(grepl("^r_region", variables(fit)))) {
    p_reg <- mcmc_plot(fit, variable = "^r_region", regex = TRUE,
                       type = "intervals", prob_outer = 0.95)
    ggplot2::ggsave(file.path(res_dir, "region_effects.png"), p_reg,
                    width = 7, height = 6, dpi = 150)
  }

}

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
# Same steps as the original M1 code, wrapped in a function so it works for
# every model: each person is predicted by (an approximation of) a model that
# did NOT see them, by resampling their draws with PSIS weights.
loo_draws <- function(fit) {
  ll    <- log_lik(fit)
  chain <- rep(seq_len(nchains(fit)), each = nrow(ll) / nchains(fit))
  psis  <- loo::psis(-ll, r_eff = loo::relative_eff(exp(ll), chain_id = chain))
  w     <- weights(psis, log = FALSE)
  yrep  <- posterior_predict(fit)
  S     <- nrow(yrep)
  # resample each column by its PSIS weights -> draws from the LOO predictive
  sapply(seq_len(ncol(yrep)), function(i)
    yrep[sample.int(S, S, replace = TRUE, prob = w[, i]), i])
}

set.seed(123)
for (name in names(fits)) {
  fit      <- fits[[name]]
  res_dir  <- here::here("output", name)
  yrep_loo <- loo_draws(fit)

  # REPORTED CALIBRATION (PPC)
  # article figure 6 comparison for interpretation (calibration curve, pred. mass, interval bands)
  p_cal <- ppc_calibration(y = dat$low3, yrep = yrep_loo,
                           interval = "confidence", prob = 0.95)
  ggplot2::ggsave(file.path(res_dir, "ppc_calibration.png"), p_cal,
                  width = 6, height = 5, dpi = 150)

  # AI -> "consider grouped calibration (ppc_calibration faceted by region / econ) since the
  # research question is about low trust within subgroups"
  # -> low-trust share per region: observed vs LOO predictive
  p_reg <- ppc_stat_grouped(dat$low3, yrep_loo, group = dat$region, stat = "mean")
  ggplot2::ggsave(file.path(res_dir, "ppc_region.png"), p_reg,
                  width = 11, height = 7, dpi = 150)
}

# PSIS_LOO -> elpd score and uncertainty
# "Report and interpret the comparison results, including uncertainty in the estimates"
# m0 = intercept-only reference (same prediction for everyone): do covariates help at all?
m0 <- fit_opts(low3 ~ 1, prior(normal(0, 2), class = "Intercept"), "m0")

all_fits <- list(m0 = m0, m1 = m1, m2 = m2, m3 = m3)
loos <- lapply(all_fits, loo, save_psis = TRUE)
loos$m3                                          # one model in detail: elpd, p_loo, Pareto k
sapply(loos, function(l) max(pareto_k_values(l)))  # worst Pareto k per model (< 0.7 is good)

cmp <- loo_compare(loos)
cmp
write.csv(as.data.frame(cmp), here::here("output", "loo_compare.csv"))

# classification scores -> "Where relevant to the research question, also assess predictive
# performance using practical measures (e.g., classification accuracy)"
# computed from the LOO predictive -> out-of-sample
y <- dat$low3
practical <- sapply(names(all_fits), function(name) {
  fit <- all_fits[[name]]
  # LOO predicted probability of low trust for each person
  p <- E_loo(posterior_epred(fit), loos[[name]]$psis_object, type = "mean",
             log_ratios = -log_lik(fit))$value
  c(accuracy = mean((p > 0.5) == y),    # share classified correctly at a 0.5 cut-off
    brier    = mean((p - y)^2))         # mean squared error of the probabilities
})
round(t(practical), 3)
mean(y == 0)                             # accuracy of always guessing "not low trust"


# PRIOR PREDICTIVE CHECKS -> group by underlying variable?
# (note: this is prior SENSITIVITY via power-scaling, not a prior predictive check)

# ---- A. power-scaling sensitivity (priorsense) for every model ----------------
for (name in names(fits)) {
  fit     <- fits[[name]]
  res_dir <- here::here("output", name)
  # population-level effects and hyperparameters (not the 18 individual region effects)
  vars  <- grep("^(b_|bsp_|bs_|sd_|sds_|simo_)", variables(fit), value = TRUE)
  psens <- powerscale_sensitivity(fit, variable = vars)
  print(psens)
  write.csv(as.data.frame(psens), file.path(res_dir, "powerscale.csv"), row.names = FALSE)
  # powerscale_plot_dens() returns a SET of panels -> print to a device
  # (plot the 6 most prior-sensitive parameters)
  png(file.path(res_dir, "powerscale_dens.png"),
      width = 10, height = 7, units = "in", res = 150)
  print(powerscale_plot_dens(fit, variable = head(psens$variable[order(-psens$prior)], 6)))
  dev.off()
}

# ---- B. refit with clearly wider / narrower priors -----------------------------
wide_b   <- c(prior(normal(0, 5), class = "Intercept"), prior(normal(0, 2.5), class = "b"))
narrow_b <- c(prior(normal(0, 1), class = "Intercept"), prior(normal(0, 0.5), class = "b"))
alt <- list(
  wide = list(
    m1 = wide_b,
    m2 = c(wide_b, prior(exponential(0.4), class = "sd")),
    m3 = c(wide_b, prior(exponential(0.4), class = "sd"), prior(normal(0, 2.5), class = "sds"),
           prior(dirichlet(1), class = "simo", coef = "moecon1"))),
  narrow = list(
    m1 = narrow_b,
    m2 = c(narrow_b, prior(exponential(2), class = "sd")),
    m3 = c(narrow_b, prior(exponential(2), class = "sd"), prior(normal(0, 0.5), class = "sds"),
           prior(dirichlet(1), class = "simo", coef = "moecon1")))
)

shift_list <- list()
for (set in names(alt)) {
  for (name in names(fits)) {
    base  <- fits[[name]]
    refit <- fit_opts(base$formula, alt[[set]][[name]], paste0(name, "_", set),
                      control = list(adapt_delta = 0.95))
    # how far did each posterior mean move, in units of its original posterior sd?
    vars <- setdiff(grep("^(b_|bsp_|bs_|sd_|sds_|simo_)", variables(base), value = TRUE),
                    "b_Intercept")
    b <- summarise_draws(subset_draws(as_draws_df(base),  variable = vars), "mean", "sd")
    a <- summarise_draws(subset_draws(as_draws_df(refit), variable = vars), "mean", "sd")
    shift_list[[paste(name, set)]] <- data.frame(
      model = name, prior = set, variable = vars,
      base_mean = round(b$mean, 2), alt_mean = round(a$mean, 2),
      shift_sd = round((a$mean - b$mean) / b$sd, 2))
  }
}
shifts <- do.call(rbind, shift_list)
write.csv(shifts, here::here("output", "prior_refits.csv"), row.names = FALSE)
head(shifts[order(-abs(shifts$shift_sd)), ], 12)   # the 12 biggest shifts

print(powerscale_plot_dens(m1))
dev.off()
powerscale_plot_ecdf(m1)

# (bayes_R2(m1) -> could include)

# classification scores (accuracy, F1, Brier, etc?) -> "Where relevant to the research question,
# also assess predictive performance using practical measures (e.g., classification accuracy) and
# discuss their practical usefulness"
# AI -> compute these from the LOO predictive (reuse m1_loo_draws) -> out-of-sample