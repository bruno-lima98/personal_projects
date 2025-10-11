# Predicting Startup Failure

## 1. Introduction
### Project Motivation
Startups are exciting, but also risky. Many fail within the first few years, and predicting which ones might struggle can help investors, founders, and stakeholders make better decisions. In this project, I explore how machine learning can be used to estimate the likelihood of a startup failing.

### Objective
The goal of this project is to build a model that predicts whether a startup will fail (shutdown) within a certain time frame after its last funding round. This is a binary classification problem, where the model outputs the probability of failure for each company.

### Problem Definition
- **Type of problem:** Binary classification .
- **Model choosed:** Logistic Regression.  
- **Target variable:** `failure` (1 if the startup fails, 0 otherwise). 
- **Scope:** Focus on building a simple, interpretable model that can provide actionable insights, not just predictions.

### Why this project?
This project allows me to apply key concepts learned in the course, such as logistic regression, feature engineering, and evaluation metrics like ROC-AUC and F1-score.  
Additionally, given my background in fintech and experience analyzing risk and performance for financial operations, I am particularly interested in how data-driven models can help assess the viability and risk of businesses. This gives me the chance to combine my professional knowledge with machine learning techniques in a meaningful way.

## 2. Data Source

For this project, I used a publicly available dataset from Kaggle.  
[Startup Failure Prediction Dataset](https://www.kaggle.com/datasets/siddarthareddyt/startup-analysis-dataset)  

### Columns Description
Here are the final columns used in the analysis and modeling:

- **category_list**: Categorical variable representing the startup's industry or sector.
- **funding_total_usd**: Numerical variable indicating the total funding in Milion USD received by the startup (log-transformed to reduce skewness and handle outliers).
- **state_code**: Categorical variable representing the US state where the startup is located.
- **funding_rounds**: Numerical variable representing the number of funding rounds the startup has completed (log-transformed).
- **number_of_investors**: Numerical variable representing the total number of investors for the startup (log-transformed).
- **startup_age**: Numerical variable representing the age of the startup in years (calculated from `founded_at` and log-transformed).
- **status_binary**: Binary target variable indicating startup failure (1 = closed, 0 = operating, acquired, or IPO).

## 3. Data Cleaning & Preprocessing

The dataset has been cleaned and prepared for analysis and modeling. Key steps include:

1. **Missing values**  
   - Dropped rows with missing values in categorical columns (`category_list`, `state_code`, `status`).  
   - Only ~45 rows removed out of ~13,000, so the data loss is minimal.

2. **Target binarization**  
   - Created `status_binary` column (1 = closed/failure, 0 = operating/acquired/IPO).

3. **Column drops**  
   - Removed `country_code` (all startups are from the USA).  
   - Removed `city` (high cardinality, not useful for modeling).

4. **Numerical features**  
   - Applied `log1p` transformation to reduce skewness in the following columns:  
     - `funding_total_usd`  
     - `funding_rounds`  
     - `number_of_investors`  
     - `startup_age`  

## 4. Exploratory Data Analysis (EDA)
