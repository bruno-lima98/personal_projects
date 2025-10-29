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
[Startup Investments Dataset](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase)  

### Columns Description
Here are the final columns used in the analysis and modeling:

- **market:** Categorical variable indicating the industry or market sector in which the startup operates.  
- **funding_total_usd:** Numerical variable showing the total amount of funding (in USD) raised by the startup.  
- **country_code:** Categorical variable representing the country where the startup is based (non-US startups were labeled as `"FGN"`).  
- **state_code:** Categorical variable representing the U.S. state where the startup is located.  
- **funding_rounds:** Number of funding rounds the startup has completed.  
- **founded_at:** Date when the company was founded.  
- **first_funding_at:** Date of the startup’s first funding round.  
- **last_funding_at:** Date of the startup’s most recent funding round.  
- **seed:** Amount of money (in USD) raised through seed funding.  
- **venture:** Amount raised through venture capital funding.  
- **equity_crowdfunding:** Amount raised through equity crowdfunding platforms.  
- **undisclosed:** Amount from undisclosed funding rounds.  
- **convertible_note:** Amount raised through convertible note financing.  
- **debt_financing:** Amount raised through debt financing.  
- **angel:** Amount raised from angel investors.  
- **grant:** Amount received through grants.  
- **private_equity:** Amount raised through private equity investments.  
- **post_ipo_equity:** Amount raised after IPO through equity offerings.  
- **post_ipo_debt:** Amount raised after IPO through debt financing.  
- **secondary_market:** Amount raised through secondary market transactions.  
- **product_crowdfunding:** Amount raised through product crowdfunding campaigns.  
- **round_A-to-H:** Amount raised across later-stage venture rounds (Series A to H).  
- **status:** Categorical variable indicating the startup’s final state (e.g., operating, acquired, closed, IPO).  
  - Used to create the binary target variable **`failed`**:
    - `1` → Startup closed (failed)  
    - `0` → Startup acquired (success)

## 3. Data Cleaning & Preprocessing

The dataset has been cleaned and prepared for analysis and modeling. Key steps include:

### 3.1 Data Validation
- Standardized all column names to lowercase and replaced spaces with underscores for consistency.  
- Removed irrelevant or high-cardinality columns, such as `name`, `homepage_url`, and `region`.  
- Filtered the dataset to include only startups with a final status of either **acquired** or **closed**.  
- Created a new binary target variable:
  - `failed = 1` → startup **closed**
  - `failed = 0` → startup **acquired**

### 3.2 Date and Currency Formatting
- Converted date columns (`founded_at`, `first_funding_at`, `last_funding_at`) into datetime format.
- Standardized the `funding_total_usd` column by removing commas, converting to lowercase, and coercing invalid values to numeric.
- Startups not based in the United States had their `state_code` and `country_code` replaced with `"FGN"` (foreign).

### 3.3 Handling Missing Values
- Removed rows missing `first_funding_at`, as this date is essential to determine the startup’s funding history.  
- Replaced missing categorical values in `market` with `"Unknown"`.  
- Created a flag variable `founded_missing` to indicate startups without founding date information.  
- Imputed missing numeric date features:
  - `founded_year` → filled with the median.
  - `founded_month` → filled with the mode.
- Replaced missing funding values with `0` to indicate no funding received.

### 3.4 Feature Engineering
- **months_to_first_funding:** calculated the number of months between the founding date and the first funding round.  
- **funding_duration_months:** calculated how long (in months) the startup received funding (difference between first and last funding dates).  
- Dropped redundant date columns (`founded_at`, `first_funding_at`, `last_funding_at`) after deriving the new time-based variables.

### 3.5 Feature Classification
At the end of the preprocessing stage, features were categorized into:

- **Categorical variables:** `market`, `country_code`, `state_code`
- **Numerical variables:** `funding_total_usd`, `funding_rounds`, `months_to_first_funding`, `funding_duration_months`, etc.

This clean and structured dataset formed the foundation for model training and evaluation.

## 4. Exploratory Data Analysis (EDA)
