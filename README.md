# Telco Customer Churn Prediction  

This repository contains a machine learning project that predicts customer churn for a telecom company. The goal is to use customer data to determine the likelihood of churn and provide insights to improve customer retention strategies.  

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Features Used](#features-used)  
3. [Dataset and Encodings](#dataset-and-encodings)  
4. [Project Workflow](#project-workflow)  
5. [Installation and Usage](#installation-and-usage)  
6. [Results](#results)  
7. [Technologies Used](#technologies-used)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Project Overview  
Customer churn prediction is critical for telecom companies to improve customer satisfaction and minimize revenue loss. This project employs supervised machine learning models to identify customers at risk of leaving the service.  

## Features Used  
The following features were selected for this analysis:  
- **tenure**: Customer tenure in months.  
- **OnlineSecurity**: Whether the customer has online security services.  
- **OnlineBackup**: Whether the customer has online backup services.  
- **DeviceProtection**: Whether the customer has device protection.  
- **TechSupport**: Whether the customer has tech support.  
- **Contract**: Type of customer contract (Month-to-month, One year, Two year).  
- **PaperlessBilling**: Whether the customer uses paperless billing.  
- **MonthlyCharges**: The amount charged monthly.  

## Dataset and Encodings  
- The dataset is based on Telco customer data.  
- Categorical variables are encoded as follows:  
  - **Yes** = 2  
  - **No** = 0  
  - **No internet service** = 1  

## Project Workflow  
1. **Data Preprocessing**: Cleaning and encoding the dataset for analysis.  
2. **Exploratory Data Analysis (EDA)**: Identifying trends and correlations in the data.  
3. **Model Training**: Using classification algorithms to predict churn.  
4. **Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, and F1-score.  

