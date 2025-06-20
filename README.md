🚗 Transport Mode Prediction
A machine learning project to predict the mode of transport used by employees (Car, Public Transport, or 2-Wheeler) based on demographic and professional features. This study evaluates multiple classification models and identifies the most significant factors influencing transport decisions.

👩‍💻 Author
Ajuna P John
Project Type: ML - Supervised Classification

🎯 Objective
To predict whether an employee uses a Car as their mode of transport and identify which variables most influence this choice. The study also compares multiple models to classify all three transport modes effectively.

📂 Dataset Overview
Source File: Cars.csv

Observations: 444

Target Variables:

Binary (CarUsage: 1 if uses Car, 0 otherwise)

Multiclass (Transport: Car, 2Wheeler, Public Transport)

🚦 Features:
Variable	Description
Age	Age of the employee
Gender	Gender (Male/Female)
Engineer	Whether the employee is an Engineer
MBA	Whether the employee has an MBA
Work.Exp	Years of work experience
Salary	Annual salary (in lakhs)
Distance	Distance to office (km)
License	Holds a valid driving license

🔍 Analysis Workflow
🧹 Data Preprocessing
Missing value imputation (e.g., MBA column)

Outlier treatment for Age, Salary, Distance, and Work Exp

Feature transformations and dummy encoding

KNN imputation and SMOTE for class balancing

📊 Exploratory Data Analysis
Boxplots, correlation matrices, and summary stats

Chi-square and correlation tests for variable relationships

🤖 Models Used
1️⃣ Binary Classification (Car vs Non-Car)
Logistic Regression

Regularized GLM (glmnet)

Accuracy: 95.45%

AUC: High sensitivity/specificity balance

Variable Importance: Age, Distance, License, MBA

2️⃣ Multiclass Classification (Car, 2Wheeler, Public Transport)
LDA (Linear Discriminant Analysis)

Accuracy: 69.7%

Penalized Discriminant Analysis (PDA)

Accuracy: 72.97%

CART (Decision Tree)

Accuracy: 61.36%

XGBoost

Accuracy: 70.45%

Multinomial Logistic Regression

Accuracy: 76.52%

Random Forest

Accuracy: 71.21%

Best balance of precision and recall across classes

💡 Key Insights
Age is the most significant predictor across models.

Employees with a driving license, higher salary, and engineer background are more likely to use a car.

Public transport is the most common overall mode.

Multinomial Logistic Regression and glmnet offered the best predictive power.

🧰 Tech Stack
Language: R

Libraries: caret, glmnet, rpart, randomForest, nnet, xgboost, MASS, mda, DMwR, ggplot2, corrplot
