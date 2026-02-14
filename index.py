# Student Performance Prediction using Python and Scikit-learn

# 1️ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("student-por.csv", sep=';')

# 3️⃣ Data Preprocessing
# Encode categorical variables
label_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# Features and target
# Predict final grade 'G3'
X = data.drop('G3', axis=1)
y = data['G3']

# Option: For classification (pass/fail), convert G3 to categories
y_class = y.apply(lambda x: 1 if x >= 10 else 0)  # 1=Pass, 0=Fail

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# 5️⃣ Build Models

# a) Random Forest Classifier (Classification)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# b) Linear Regression (Regression on G3 numeric grades)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
y_pred_r = lr_model.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
print("Linear Regression MSE:", mse)

# 6️⃣ Feature Importance (Random Forest)
importances = rf_model.feature_importances_
features = data.drop('G3', axis=1).columns

feat_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)
print(feat_importance.head(10))

# Optional: Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance.head(10))
plt.title("Top 10 Feature Importances")
plt.show()
