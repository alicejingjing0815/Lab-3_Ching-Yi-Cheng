#Lab 3: Linear Regression: Model Specification, Comparison, and Interpretation
#Complaint Handling, Learning Opportunities, and Performance-Based Raises as Predictors of Overall Workplace Ratings

## 1. Load packages and prepare the setup
library(carData)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom)
library(performance)
library(gt)

### Keep the same complete-case sample for every model
dat <- attitude |>
  select(rating, complaints, learning, raises)

n <- nrow(dat)

## 2. Explore the data
### Descriptive statistics
#| label: descriptive-table
#| tbl-cap: "Descriptive statistics for workplace attitude variables"
descriptives <- dat |>
  summarise(
    across(
      c(rating, complaints, learning, raises),
      list(mean = mean, sd = sd, min = min, max = max)
    )
  ) |>
  pivot_longer(
    everything(),
    names_to = c("variable", ".value"),
    names_sep = "_"
  )

descriptives |>
  gt() |>
  fmt_number(columns = where(is.numeric), decimals = 2)


### Distribution of rating
ggplot(dat, aes(x = rating)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 15,
    fill = "#669BBC",
    color = "white",
    alpha = 0.8
  ) +
  geom_density(
    linewidth = 0.9,
    color = "#9A3038"
  ) +
  labs(
    x = "Rating score",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)


### Relationships between predictors and rating
ggplot(dat, aes(x = complaints, y = rating)) +
  geom_point(alpha = 0.75) +
  geom_smooth(
    method = "lm",
    se = TRUE,
    color = "#27638A"
  ) +
  labs(
    x = "Complaint handling (% favorable)",
    y = "Overall workplace rating (% favorable)"
  ) +
  theme_minimal(base_size = 12)

### Relationship between learning and rating

#| label: learning-scatterplot
#| fig-cap: "Relationship between learning and rating"
ggplot(dat, aes(x = learning, y = rating)) +
  geom_point(alpha = 0.75) +
  geom_smooth(
    method = "lm",
    se = TRUE,
    color = "#27638A"
  ) +
  labs(
    x = "Learning opportunities (% favorable)",
    y = "Overall workplace rating (% favorable)"
  ) +
  theme_minimal(base_size = 12)


## 3. Specify and estimate competing models
# Null: Intercept-only model
mod0 <- lm(rating ~ 1, data = dat)

# Complaints model
mod1 <- lm(rating ~ complaints, data = dat)

# Complaints + learning model
mod2 <- lm(rating ~ complaints + learning, data = dat)

# Expanded model
mod3 <- lm(rating ~ complaints + learning + raises, data = dat)


### Inspecting the models
summary(mod0)
summary(mod1)
summary(mod2)
summary(mod3)

confint(mod3)


## 4. Compare the models
### 4.1. Conduct nested-model comparisons
#| label: nested-tests
tests <- anova(mod0, mod1, mod2, mod3)
tests

#| label: prepare-nested-tests
tests <- tests |>
  select(Res.Df, Df, F, `Pr(>F)`) |>
  rename(
    Residual.df = Res.Df,
    Nested.Df = Df,
    Nested.F.stat = F,
    Nested.p.value = `Pr(>F)`
  )


### 4.2. Extract overall model performance
#| label: extract-model-metrics
glance0 <- glance(mod0)
glance1 <- glance(mod1)
glance2 <- glance(mod2)
glance3 <- glance(mod3)


#| label: combine-model-metrics
mods <- c(
  "M0: Null",
  "M1: Complaints",
  "M2: + Learning",
  "M3: + Raises"
)

glances <- rbind(
  glance0,
  glance1,
  glance2,
  glance3
)


### 4.3. Combine nested tests and model performance
#| label: combine-comparisons
comparison_table <- cbind(
  Model = mods,
  glances,
  tests
)


#| label: prepare-comparison-table
comparison_table <- comparison_table |>
  mutate(
    Delta.R2 = r.squared - lag(r.squared),
    
    RMSE = c(
      sqrt(mean(residuals(mod0)^2)),
      sqrt(mean(residuals(mod1)^2)),
      sqrt(mean(residuals(mod2)^2)),
      sqrt(mean(residuals(mod3)^2))
    )
  ) |>
  select(
    Model,
    r.squared,
    Delta.R2,
    adj.r.squared,
    RMSE,
    Residual.df,
    Nested.Df,
    Nested.F.stat,
    Nested.p.value
  )

### 4.4. Display the combined model-comparison table
#| label: model-comparison-table
#| tbl-cap: "Overall model performance and incremental nested-model comparisons"
comparison_table |>
  gt() |>
  cols_label(
    Model = "Model",
    r.squared = "R²",
    Delta.R2 = "ΔR²",
    adj.r.squared = "Adjusted R²",
    RMSE = "RMSE",
    Residual.df = "Residual df",
    Nested.Df = "Nested df",
    Nested.F.stat = "Nested F",
    Nested.p.value = "Nested p"
  ) |>
  fmt_number(
    columns = c(
      r.squared,
      Delta.R2,
      adj.r.squared,
      RMSE,
      Nested.F.stat
    ),
    decimals = 3
  ) |>
  fmt(
    columns = Nested.p.value,
    fns = function(x) {
      ifelse(
        is.na(x),
        "—",
        ifelse(x < .001, "< .001", sprintf("%.3f", x))
      )
    }
  ) |>
  sub_missing(
    columns = c(Delta.R2, Nested.Df, Nested.F.stat),
    missing_text = "—"
  )

### 4.5. Additional performance comparison
#| label: performance-comparison
#| results: hide
performance_comparison <- compare_performance(
  mod0,
  mod1,
  mod2,
  mod3,
  rank = TRUE
)

performance_comparison
plot(performance_comparison)


## 5. Select a provisional model
#| label: select-model
selected_row <- comparison_table |>
  filter(Model == "M2: + Learning")

selected_name <- selected_row$Model
selected_model <- mod2


## 6. Evaluate regression diagnostics
#| label: diagnostics
#| fig-cap: "Regression diagnostic plots for M2"
#| fig-width: 9
#| fig-height: 8

op <- par(mfrow = c(2, 2))
plot(selected_model)
par(op)


#| label: collinearity
check_collinearity(selected_model)


## 7. Select and report the preferred model
### 7.1. Model selection
#| label: final-model-selection
selected_model <- mod2

selected_row <- comparison_table |>
  filter(Model == "M2: + Learning")

selected_name <- selected_row$Model


### 7.2. Regression coefficient table
#| label: coefficient-table
#| tbl-cap: "Unstandardized regression coefficients for M2"
coefficient_results <- tidy(
  selected_model,
  conf.int = TRUE,
  conf.level = 0.95
)

coefficient_results |>
  select(
    term, estimate, std.error,
    conf.low, conf.high, p.value
  ) |>
  gt() |>
  cols_label(
    term = "Predictor",
    estimate = "Estimate",
    std.error = "SE",
    conf.low = "95% CI lower",
    conf.high = "95% CI upper",
    p.value = "p"
  ) |>
  fmt_number(
    columns = c(
      estimate, std.error,
      conf.low, conf.high
    ),
    decimals = 3
  ) |>
  fmt(
    columns = p.value,
    fns = function(x) {
      ifelse(x < .001, "< .001", sprintf("%.3f", x))
    }
  )
