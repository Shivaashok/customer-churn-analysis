PROJECT TITLE: Telecom Customer Churn Analysis using Python

DESCRIPTION:
This project analyzes customer behavior in a telecom company to predict whether a customer will churn (leave) or stay. Using Python-based data analysis and machine learning, the project identifies key factors that influence customer retention. The goal is to help telecom companies take proactive steps to reduce churn and improve customer satisfaction.

OBJECTIVE:
To build a predictive model that can accurately identify customers likely to churn based on demographic, service usage, and account data.

DATASET:
WA_Fn-UseC_-Telco-Customer-Churn.csv
The dataset contains 7,043 customer records with attributes such as gender, tenure, contract type, monthly charges, internet service, and churn status.

TOOLS AND LIBRARIES USED:

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

Jupyter Notebook

STEPS INVOLVED:

Data Collection and Cleaning

Loaded dataset and handled missing values

Converted categorical values into numerical codes

Cleaned “TotalCharges” column

Exploratory Data Analysis (EDA)

Analyzed churn distribution

Visualized relationships between churn and key features like tenure, contract, and internet type

Model Building

Used Logistic Regression as the predictive model

Split data into training and testing sets (80:20)

Model Evaluation

Achieved 90% accuracy on the test dataset

Evaluated results using a confusion matrix and classification report

Insights and Recommendations

Customers with short tenure and month-to-month contracts are more likely to churn

Fiber optic users show higher churn rates

Customers with longer contracts and lower monthly charges are more loyal

RESULTS:
Model Accuracy: 90%
The model effectively predicts churn and highlights major influencing factors like tenure, contract type, and internet service.

CONCLUSION:
The project successfully demonstrates how machine learning can be applied to predict telecom customer churn. By understanding the drivers behind churn, telecom companies can design targeted offers, loyalty programs, and service improvements to enhance customer retention.
