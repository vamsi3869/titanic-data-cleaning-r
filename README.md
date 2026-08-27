# Titanic Dataset — Data Cleaning & Preliminary Analysis (R)

**Week 1 Task — Virtual R Data Analyst Internship**

This repository contains the data cleaning, preprocessing, and exploratory
analysis performed on the Titanic passenger dataset using R (tidyverse /
ggplot2). It covers missing value treatment, outlier detection,
transformation/normalization, categorical encoding, and initial exploratory
insights into passenger survival patterns.

## 📂 Repository Contents

| File | Description |
|---|---|
| `titanic_analysis.R` | Complete, self-contained R script covering the full pipeline: load → inspect → clean → transform → encode → analyze → visualize |
| `titanic_raw.csv` | Raw Titanic passenger dataset (891 rows, 12 columns) used as input |
| `Titanic_Data_Cleaning_Analysis_Report.docx` | Full written report with embedded R code, outputs, and charts *(if included)* |

## 📊 Dataset

- **Source:** [Data Science Dojo — Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (mirror of the original Kaggle "Titanic: Machine Learning from Disaster" training set)
- **Rows:** 891 passengers
- **Columns:** 12 (mix of numeric, categorical, and text fields)
- **Missing data:** Age (177 missing, 19.9%), Cabin (687 missing, 77.1%), Embarked (2 missing, 0.2%)

## 🧹 Data Cleaning Steps

1. **Initial inspection** — `str()`, `summary()`, `glimpse()` to understand structure and missingness
2. **Missing value treatment**
   - `Embarked`: imputed with the mode (most frequent port)
   - `Age`: imputed with the median age within each `Pclass × Sex` subgroup
   - `Cabin`: too sparse (77% missing) to impute meaningfully — engineered into a `HasCabin` binary indicator and a `Deck` feature instead
3. **Outlier detection** — IQR method applied to `Age` and `Fare`; `Fare` outliers (13.0% of records) capped (winsorized) at the upper IQR bound
4. **Transformation & normalization** — `log(1+Fare)` transform to correct heavy right-skew (skewness 4.79 → 0.39); min-max and z-score scaling applied to key numeric variables
5. **Feature engineering** — `FamilySize`, `IsAlone`, and `Title` (extracted from passenger name) added
6. **Encoding** — `Sex` label-encoded; `Embarked` and `Pclass` one-hot encoded

## 🔍 Key Findings

| Insight | Value |
|---|---|
| Overall survival rate | 38.3% |
| Female survival rate | 73.9% |
| Male survival rate | 18.9% |
| 1st class survival rate | 63.0% |
| 2nd class survival rate | 47.3% |
| 3rd class survival rate | 24.0% |
| Strongest correlate of survival | Sex (r = 0.54) |

Sex was the dominant predictor of survival, and passenger class amplified
that effect substantially — first-class women survived at ~97%, while
third-class men survived at only ~14%.

## ▶️ How to Run

```r
# Install required packages (one-time)
install.packages(c("tidyverse", "janitor", "corrplot", "scales", "e1071", "fastDummies"))

# Run the full pipeline
source("titanic_analysis.R")
```

The script reads the dataset directly from the public URL, so an internet
connection is required, or you can point it at the local `titanic_raw.csv`
included in this repo instead.

## 🛠️ Tools Used

- **R** (tidyverse, ggplot2, dplyr, janitor, corrplot, scales, e1071, fastDummies)

---
*Submitted as part of the Week 1 Data Cleaning and Preliminary Analysis task.*
