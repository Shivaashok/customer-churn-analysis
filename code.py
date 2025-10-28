# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# 3️⃣ Data Cleaning
# Remove customerID column (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric (it sometimes has spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 4️⃣ Convert categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

# Encode Yes/No columns to 1/0
binary_cols = df.columns[df.isin(['Yes', 'No']).any()]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Encode remaining categorical variables (like gender, InternetService, Contract)
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:", df.shape)

# 5️⃣ Basic EDA
print("\nChurn distribution:")
print(df['Churn'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Count")
plt.show()

# Check correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 6️⃣ Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8️⃣ Evaluate Model
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9️⃣ Feature Importance
importance = pd.Series(model.coef_[0], index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)
bottom_features = importance.sort_values().head(10)

plt.figure(figsize=(10,5))
top_features.plot(kind='barh', color='green')
plt.title("Top 10 Factors Increasing Churn Probability")
plt.show()

plt.figure(figsize=(10,5))
bottom_features.plot(kind='barh', color='red')
plt.title("Top 10 Factors Decreasing Churn Probability")
plt.show()

# 10️⃣ Conclusion
print("✅ Model completed successfully!")
print("Most important features influencing churn:")
print(top_features)
