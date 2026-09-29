## ============================================================
## Titanic Dataset: Statistical Analysis & Predictive Modeling
## Week 3 Task — Data Analysis Internship
## ============================================================

library(tidyverse)
library(caret)       # train/test split, cross-validation, confusion matrix
library(car)         # vif()
library(pROC)        # ROC curve / AUC
library(broom)        # tidy model summaries

## ---- 1. Load the cleaned dataset (from Week 1) ----
titanic <- read_csv("titanic_clean.csv", show_col_types = FALSE)

titanic <- titanic %>%
  mutate(
    Survived = factor(Survived, labels = c("Died", "Survived")),
    Embarked = factor(Embarked)
  )

## ============================================================
## PART A: EXPLORATORY STATISTICAL ANALYSIS / HYPOTHESIS TESTING
## ============================================================

## ---- A1. Normality tests (Shapiro-Wilk) ----
shapiro.test(titanic$Age)
shapiro.test(titanic$Fare)
## Both p < .001 -> reject normality for Age and (especially) Fare.
## This justifies using non-parametric/robust methods where relevant and
## motivates the earlier log-transform of Fare (Week 1).

## Visual check: Q-Q plots
qqnorm(titanic$Age, main = "Q-Q Plot: Age"); qqline(titanic$Age, col = "#e07a5f")
qqnorm(titanic$Fare, main = "Q-Q Plot: Fare"); qqline(titanic$Fare, col = "#e07a5f")

## ---- A2. Hypothesis test: Does Age differ by survival status? ----
## H0: mean Age is equal for survivors and non-survivors
## H1: mean Age differs
leveneTest(Age ~ Survived, data = titanic)          # test equal variances
t.test(Age ~ Survived, data = titanic, var.equal = FALSE)  # Welch's t-test

## ---- A3. Hypothesis test: Does Fare differ by survival status? ----
## H0: mean Fare is equal for survivors and non-survivors
t.test(Fare ~ Survived, data = titanic, var.equal = FALSE)

## ---- A4. Chi-square tests of independence (categorical predictors) ----
## H0: Survival is independent of Sex / Pclass / Embarked
chisq.test(table(titanic$Sex, titanic$Survived))
chisq.test(table(titanic$Pclass, titanic$Survived))
chisq.test(table(titanic$Embarked, titanic$Survived))

## ---- A5. Correlation test: Age vs Fare ----
## H0: true correlation is 0
cor.test(titanic$Age, titanic$Fare, method = "pearson")

## ============================================================
## PART B: PREDICTIVE MODEL — LOGISTIC REGRESSION (CLASSIFICATION)
## ============================================================
## Survived is binary, so logistic regression is the appropriate GLM.

titanic_model <- titanic %>%
  mutate(Survived_num = if_else(Survived == "Survived", 1L, 0L)) %>%
  select(Survived, Survived_num, Pclass, Sex_encoded, Age, Fare, FamilySize, Embarked)

## ---- B1. Train/test split (70/30), stratified on the outcome ----
set.seed(42)
split_idx <- createDataPartition(titanic_model$Survived, p = 0.7, list = FALSE)
train_data <- titanic_model[split_idx, ]
test_data  <- titanic_model[-split_idx, ]

## ---- B2. Fit the logistic regression model ----
logit_model <- glm(
  Survived ~ Pclass + Sex_encoded + Age + Fare + FamilySize + Embarked,
  data = train_data, family = binomial(link = "logit")
)
summary(logit_model)
tidy(logit_model, exponentiate = TRUE, conf.int = TRUE)  # odds ratios + CIs

## Model fit statistics
logit_model$null.deviance
logit_model$deviance
AIC(logit_model)

## ---- B3. Multicollinearity check (Variance Inflation Factor) ----
vif(logit_model)
## Rule of thumb: VIF > 5 (some use 10) signals problematic collinearity.

## ---- B4. Predict on the test set & build the confusion matrix ----
test_data$prob <- predict(logit_model, newdata = test_data, type = "response")
test_data$pred <- factor(if_else(test_data$prob >= 0.5, "Survived", "Died"),
                          levels = levels(test_data$Survived))

confusionMatrix(test_data$pred, test_data$Survived, positive = "Survived")

## ---- B5. ROC curve and AUC ----
roc_obj <- roc(test_data$Survived, test_data$prob, levels = c("Died", "Survived"))
plot(roc_obj, main = paste0("ROC Curve (AUC = ", round(auc(roc_obj), 3), ")"))
auc(roc_obj)

## ---- B6. 5-fold cross-validation (robustness check) ----
train_control <- trainControl(method = "cv", number = 5)
cv_model <- train(
  Survived ~ Pclass + Sex_encoded + Age + Fare + FamilySize + Embarked,
  data = titanic_model, method = "glm", family = "binomial",
  trControl = train_control
)
print(cv_model)
cv_model$resample   # per-fold accuracy

## ============================================================
## PART C: DIAGNOSTIC PLOTS
## ============================================================

## C1. Residuals vs fitted (deviance residuals)
plot(logit_model, which = 1)

## C2. Confusion matrix heatmap
cm <- confusionMatrix(test_data$pred, test_data$Survived, positive = "Survived")
cm_table <- as.data.frame(cm$table)
ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "#3d5a80") +
  labs(title = "Confusion Matrix — Test Set") +
  theme_minimal()

## C3. Odds ratio plot
tidy(logit_model, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "#3d5a80") +
  geom_hline(yintercept = 1, linetype = "dashed") +
  coord_flip() +
  labs(title = "Odds Ratios by Predictor", x = NULL, y = "Odds Ratio") +
  theme_minimal()

## C4. Cross-validation accuracy by fold
ggplot(cv_model$resample, aes(x = Resample, y = Accuracy)) +
  geom_col(fill = "#81b29a") +
  geom_hline(yintercept = mean(cv_model$resample$Accuracy), color = "#e07a5f", linetype = "dashed") +
  labs(title = "5-Fold Cross-Validation Accuracy", x = NULL, y = "Accuracy") +
  theme_minimal()
