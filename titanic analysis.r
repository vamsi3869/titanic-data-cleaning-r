## ============================================================
## Titanic Dataset: Data Cleaning, Preprocessing & Preliminary
## Exploratory Analysis
## Week 1 Task — Data Analysis Internship
## ============================================================

## ---- 0. Setup ----
library(tidyverse)   # dplyr, ggplot2, readr, stringr, tidyr
library(janitor)     # clean_names(), tabyl()
library(corrplot)    # correlation heatmap
library(scales)      # percent axis labels

## ---- 1. Load the data ----
# Public dataset: Titanic passenger manifest (Data Science Dojo mirror of the
# original Kaggle "Titanic: Machine Learning from Disaster" training set)
titanic <- read_csv(
  "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
  show_col_types = FALSE
)

## ---- 2. Initial inspection ----
str(titanic)
summary(titanic)
dim(titanic)
glimpse(titanic)

## Count and percentage of missing values per column
missing_report <- titanic %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  mutate(pct_missing = round(100 * n_missing / nrow(titanic), 1)) %>%
  filter(n_missing > 0) %>%
  arrange(desc(n_missing))
print(missing_report)

## ---- 3. Handling missing values ----

## 3a. Embarked: only 2 missing -> impute with the mode
embarked_mode <- titanic %>%
  count(Embarked) %>%
  filter(!is.na(Embarked)) %>%
  slice_max(n, n = 1) %>%
  pull(Embarked)

titanic <- titanic %>%
  mutate(Embarked = if_else(is.na(Embarked), embarked_mode, Embarked))

## 3b. Age: 177 missing (~20%) -> impute with the median Age within each
## Pclass x Sex subgroup (more faithful than one single global median,
## since age differs systematically by class and by sex on this dataset)
titanic <- titanic %>%
  group_by(Pclass, Sex) %>%
  mutate(Age = if_else(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

## 3c. Cabin: 687 missing (77%) -> too sparse to impute meaningfully.
## Engineer a binary indicator and a Deck feature instead of imputing values.
titanic <- titanic %>%
  mutate(
    HasCabin = if_else(is.na(Cabin), 0L, 1L),
    Deck     = if_else(is.na(Cabin), "Unknown", str_sub(Cabin, 1, 1))
  )

## Confirm Age/Embarked no longer have missing values
colSums(is.na(titanic %>% select(Age, Embarked)))

## ---- 4. Outlier detection (IQR method) ----
iqr_bounds <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  c(lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr)
}

age_bounds  <- iqr_bounds(titanic$Age)
fare_bounds <- iqr_bounds(titanic$Fare)

n_age_outliers  <- sum(titanic$Age  < age_bounds["lower"]  | titanic$Age  > age_bounds["upper"])
n_fare_outliers <- sum(titanic$Fare < fare_bounds["lower"] | titanic$Fare > fare_bounds["upper"])

cat("Age outliers:", n_age_outliers, " | Fare outliers:", n_fare_outliers, "\n")

## Boxplots to visualise outliers
ggplot(titanic, aes(y = Age)) +
  geom_boxplot(fill = "#81b29a") +
  labs(title = "Age — Outlier Check") +
  theme_minimal()

ggplot(titanic, aes(y = Fare)) +
  geom_boxplot(fill = "#e07a5f") +
  labs(title = "Fare — Outlier Check") +
  theme_minimal()

## Cap (winsorize) Fare at the upper IQR bound
titanic <- titanic %>%
  mutate(Fare_capped = pmin(Fare, fare_bounds["upper"]))

## Fare is heavily right-skewed (a small number of very high-fare tickets) ->
## apply a log(1+x) transform, standard for monetary/price variables
titanic <- titanic %>%
  mutate(Fare_log = log1p(Fare))

cat("Skewness before log transform:", e1071::skewness(titanic$Fare), "\n")
cat("Skewness after  log transform:", e1071::skewness(titanic$Fare_log), "\n")

## ---- 5. Feature engineering ----
titanic <- titanic %>%
  mutate(
    FamilySize = SibSp + Parch + 1,
    IsAlone    = if_else(FamilySize == 1, 1L, 0L),
    Title      = str_extract(Name, "(?<=,\\s)[^\\.]+(?=\\.)"),
    Title      = case_match(Title,
                             c("Mlle", "Ms") ~ "Miss",
                             "Mme" ~ "Mrs",
                             c("Lady","Countess","Capt","Col","Don","Dr",
                               "Major","Rev","Sir","Jonkheer","Dona") ~ "Rare",
                             .default = Title)
  )

## ---- 6. Normalization & standardization of numeric variables ----
min_max <- function(x) (x - min(x)) / (max(x) - min(x))

titanic <- titanic %>%
  mutate(
    Age_norm        = min_max(Age),
    Fare_log_norm   = min_max(Fare_log),
    FamilySize_norm = min_max(FamilySize),
    Age_z           = as.numeric(scale(Age)),
    Fare_log_z      = as.numeric(scale(Fare_log))
  )

## ---- 7. Encoding categorical variables ----
titanic <- titanic %>%
  mutate(Sex_encoded = if_else(Sex == "female", 1L, 0L))

## One-hot encode Embarked and Pclass (drop first level to avoid the dummy trap)
titanic_model <- titanic %>%
  mutate(Pclass_f = factor(Pclass), Embarked_f = factor(Embarked)) %>%
  fastDummies::dummy_cols(select_columns = c("Embarked_f", "Pclass_f"),
                           remove_first_dummy = TRUE, remove_selected_columns = TRUE)

## ---- 8. Exploratory analysis: descriptive statistics ----
summary(titanic %>% select(Age, Fare, FamilySize, SibSp, Parch))

titanic %>%
  summarise(
    survival_rate = mean(Survived),
    n = n()
  )

titanic %>% group_by(Sex)      %>% summarise(survival_rate = mean(Survived), n = n())
titanic %>% group_by(Pclass)   %>% summarise(survival_rate = mean(Survived), n = n())
titanic %>% group_by(Embarked) %>% summarise(survival_rate = mean(Survived), n = n())
titanic %>% group_by(IsAlone)  %>% summarise(survival_rate = mean(Survived), n = n())

## ---- 9. Correlation matrix ----
num_vars <- titanic %>%
  transmute(Survived, Pclass, Age, SibSp, Parch, Fare, FamilySize,
            Sex_encoded, HasCabin)

corr_matrix <- cor(num_vars, use = "complete.obs")
round(corr_matrix, 2)

corrplot(corr_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix — Numeric & Encoded Variables",
         mar = c(0, 0, 2, 0))

## ---- 10. Key visualisations ----

## Missingness overview
missing_report %>%
  ggplot(aes(x = reorder(column, -n_missing), y = n_missing)) +
  geom_col(fill = "#e07a5f") +
  geom_text(aes(label = n_missing), vjust = -0.4) +
  labs(title = "Missing Values by Column (Before Cleaning)",
       x = NULL, y = "Missing count") +
  theme_minimal()

## Age distribution before vs after imputation would be compared by keeping
## a copy of the raw column prior to Step 3b (omitted here for brevity —
## see the report for the before/after comparison figure)

## Fare distribution: raw vs log-transformed
ggplot(titanic, aes(x = Fare)) +
  geom_histogram(bins = 40, fill = "#f2cc8f") +
  labs(title = "Fare Distribution (Raw)") +
  theme_minimal()

ggplot(titanic, aes(x = Fare_log)) +
  geom_histogram(bins = 40, fill = "#f2cc8f") +
  labs(title = "log(1 + Fare) Distribution") +
  theme_minimal()

## Survival rate by sex and by class
ggplot(titanic, aes(x = Sex, y = Survived)) +
  stat_summary(fun = mean, geom = "bar", fill = "#81b29a") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate by Sex", y = "Survival rate") +
  theme_minimal()

ggplot(titanic, aes(x = factor(Pclass), y = Survived)) +
  stat_summary(fun = mean, geom = "bar", fill = "#3d5a80") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate by Passenger Class", x = "Pclass", y = "Survival rate") +
  theme_minimal()

## Survival by class, split by sex
ggplot(titanic, aes(x = factor(Pclass), y = Survived, fill = Sex)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate by Passenger Class and Sex",
       x = "Pclass", y = "Survival rate") +
  theme_minimal()

## Survival by family size
titanic %>%
  group_by(FamilySize) %>%
  summarise(survival_rate = mean(Survived)) %>%
  ggplot(aes(x = factor(FamilySize), y = survival_rate)) +
  geom_col(fill = "#3d5a80") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate by Family Size", x = "Family size (incl. self)",
       y = "Survival rate") +
  theme_minimal()

## ---- 11. Save the cleaned dataset ----
write_csv(titanic, "titanic_clean.csv")
