## ============================================================
## Titanic Dataset: Data Visualization & Insight Communication
## Week 2 Task — Data Analysis Internship
## ============================================================

library(tidyverse)
library(scales)
library(viridis)

## ---- 1. Load the cleaned dataset (from Week 1) ----
titanic <- read_csv("titanic_clean.csv", show_col_types = FALSE)

## Create age groups for trend visualizations
titanic <- titanic %>%
  mutate(AgeGroup = cut(Age,
                         breaks = c(0, 12, 18, 30, 45, 60, 80),
                         labels = c("0-12 (Child)", "13-18 (Teen)", "19-30 (Young Adult)",
                                    "31-45 (Adult)", "46-60 (Middle Age)", "61-80 (Senior)"),
                         include.lowest = TRUE))

## ---- 2. BAR CHART: Passenger count by class ----
ggplot(titanic, aes(x = factor(Pclass), fill = factor(Pclass))) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("#3d5a80", "#81b29a", "#e07a5f"), guide = "none") +
  labs(title = "Passenger Count by Class", x = "Passenger Class", y = "Number of Passengers") +
  theme_minimal()

## ---- 3. HISTOGRAM: Age distribution ----
ggplot(titanic, aes(x = Age)) +
  geom_histogram(bins = 30, fill = "#3d5a80", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(Age)), color = "#e07a5f", linetype = "dashed", linewidth = 1) +
  geom_vline(aes(xintercept = median(Age)), color = "#f2cc8f", linetype = "dashed", linewidth = 1) +
  labs(title = "Distribution of Passenger Age", x = "Age (years)", y = "Count") +
  theme_minimal()

## ---- 4. SCATTER PLOT: Age vs Fare, colored by survival ----
ggplot(titanic, aes(x = Age, y = Fare, color = factor(Survived))) +
  geom_point(alpha = 0.55, size = 2) +
  scale_color_manual(values = c("0" = "#e07a5f", "1" = "#3d5a80"),
                      labels = c("Did not survive", "Survived"), name = NULL) +
  labs(title = "Age vs. Fare, Colored by Survival Outcome", x = "Age (years)", y = "Fare (£)") +
  theme_minimal()

## ---- 5. LINE CHART: Survival rate trend across age groups ----
titanic %>%
  group_by(AgeGroup) %>%
  summarise(survival_rate = mean(Survived)) %>%
  ggplot(aes(x = AgeGroup, y = survival_rate, group = 1)) +
  geom_line(color = "#3d5a80", linewidth = 1.2) +
  geom_point(color = "#3d5a80", size = 3) +
  geom_text(aes(label = percent(survival_rate, accuracy = 1)), vjust = -1) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate Trend Across Age Groups", x = "Age Group", y = "Survival Rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

## ---- 6. BOXPLOT: Fare distribution by class ----
ggplot(titanic, aes(x = factor(Pclass), y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  scale_fill_manual(values = c("#3d5a80", "#81b29a", "#e07a5f"), guide = "none") +
  labs(title = "Fare Distribution by Passenger Class", x = "Passenger Class", y = "Fare (£)") +
  theme_minimal()

## ---- 7. STACKED BAR: Survival count by class and sex ----
titanic %>%
  mutate(Survived_f = factor(Survived, labels = c("Did not survive", "Survived"))) %>%
  ggplot(aes(x = interaction(Pclass, Sex), fill = Survived_f)) +
  geom_bar() +
  scale_fill_manual(values = c("#e07a5f", "#3d5a80"), name = NULL) +
  labs(title = "Survival Count by Class and Sex", x = "(Class, Sex)", y = "Number of Passengers") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

## ---- 8. VIOLIN PLOT: Age distribution by survival status ----
titanic %>%
  mutate(Survived_f = factor(Survived, labels = c("Did not survive", "Survived"))) %>%
  ggplot(aes(x = Survived_f, y = Age, fill = Survived_f)) +
  geom_violin() +
  scale_fill_manual(values = c("#e07a5f", "#3d5a80"), guide = "none") +
  labs(title = "Age Distribution by Survival Status", x = NULL, y = "Age (years)") +
  theme_minimal()

## ---- 9. HEATMAP: Survival rate by class x age group ----
titanic %>%
  group_by(Pclass, AgeGroup) %>%
  summarise(survival_rate = mean(Survived), .groups = "drop") %>%
  ggplot(aes(x = AgeGroup, y = factor(Pclass), fill = survival_rate)) +
  geom_tile() +
  geom_text(aes(label = percent(survival_rate, accuracy = 1)), color = "black") +
  scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                        midpoint = 0.5, name = "Survival\nrate") +
  labs(title = "Survival Rate by Class and Age Group", x = "Age Group", y = "Passenger Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

## ---- 10. GROUPED BAR: Survival rate by embarkation port ----
titanic %>%
  mutate(Port = recode(Embarked, C = "Cherbourg", Q = "Queenstown", S = "Southampton")) %>%
  group_by(Port) %>%
  summarise(survival_rate = mean(Survived)) %>%
  arrange(desc(survival_rate)) %>%
  ggplot(aes(x = reorder(Port, -survival_rate), y = survival_rate, fill = Port)) +
  geom_col() +
  geom_text(aes(label = percent(survival_rate, accuracy = 1)), vjust = -0.5) +
  scale_fill_manual(values = c("#3d5a80", "#81b29a", "#e07a5f"), guide = "none") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Survival Rate by Port of Embarkation", x = NULL, y = "Survival Rate") +
  theme_minimal()

## ---- 11. DENSITY PLOT: Fare (log scale) by class ----
titanic %>%
  mutate(Fare_log = log1p(Fare)) %>%
  ggplot(aes(x = Fare_log, fill = factor(Pclass))) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = c("#3d5a80", "#81b29a", "#e07a5f"), name = "Class") +
  labs(title = "Fare Density by Passenger Class (log scale)", x = "log(1 + Fare)", y = "Density") +
  theme_minimal()
