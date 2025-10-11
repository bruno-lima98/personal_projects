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
[Startup Failure Prediction Dataset](https://www.kaggle.com/datasets/dagloxkankwanda/startup-failures/data)  

### Columns Description
- **Startup_Name**: Identifier of the startup.  
- **Industry**: Industry or sector in which the startup operates.  
- **Startup_Age**: Age of the startup in years.  
- **Funding_Amount**: Total funding received by the startup in USD.  
- **Number_of_Founders**: Number of founders of the startup.  
- **Founder_Experience**: Work experience of the founders.  
- **Employees_Count**: Number of employees in the startup.  
- **Revenue**: Annual revenue of the startup in USD.  
- **Burn_Rate**: Monthly cash burn rate of the startup in USD.  
- **Market_Size**: Estimated size of the target market in categories.  
- **Business_Model**: Type of business model.. 
- **Product_Uniqueness_Score**: A score representing how unique or differentiated the product is (1-10).  
- **Customer_Retention_Rate**: Percentage of customers retained over a given period.  
- **Marketing_Expense**: Annual spending on marketing in USD.  
- **Startup_Status**: 
    - Target variable.
    - 1: Successful of the Startup.
    - 0: The startup failed.
