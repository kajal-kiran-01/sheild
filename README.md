# Phishing Detection using XGBoost and SMOTE

## Overview
This project aims to detect phishing websites using machine learning. The dataset is balanced using SMOTE (Synthetic Minority Over-sampling Technique) to improve classification performance. The final model is trained using XGBoost and achieves high accuracy in detecting phishing websites.

## Dataset
The dataset is assumed to be in CSV format and contains features that help distinguish between phishing and legitimate websites. The target variable (`class`) is labeled as:
- `-1` for phishing websites
- `1` for legitimate websites

## Installation and Requirements
To run this project, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost pickle5
```

## Steps in the Model Pipeline
### 1. Load the Dataset
The dataset is loaded from a CSV file and split into features (`X`) and target (`y`).

### 2. Balance the Dataset with SMOTE
Since phishing datasets often have imbalanced class distributions, SMOTE is used to oversample the minority class to create a balanced dataset.

### 3. Split the Data
The resampled dataset is split into training and testing sets (80/20 split).

### 4. Train XGBoost Model
The `XGBClassifier` is used to train the model with 100 estimators and a learning rate of 0.1.

### 5. Evaluate the Model
The trained model is evaluated using:
- Accuracy Score
- Precision, Recall, and F1-score

### 6. Save the Model
The trained model is saved as a pickle file (`phishing_model_v2.pkl`) for future use.

## Code Execution
Run the following script to train and evaluate the model:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("phishing.csv")

# Separate features and target
X = df.drop(columns=["class"])
y = df["class"]

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert -1 to 0 in labels
y_train = y_train.replace(-1, 0)
y_test = y_test.replace(-1, 0)

# Train XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print results
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("phishing_model_v2.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as phishing_model_v2.pkl")
```

## Results
The model achieves:
- **Accuracy**: ~96.5%
- **Precision, Recall, F1-score**: High performance in phishing detection.

## Usage
You can load the trained model and use it for predictions:

```python
with open("phishing_model_v2.pkl", "rb") as file:
    model = pickle.load(file)

# Example prediction
sample_data = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample_data)
print("Prediction (0 = Phishing, 1 = Legitimate):", prediction[0])
```

## Conclusion
This project demonstrates how machine learning can effectively detect phishing websites. The use of SMOTE improves model performance, and XGBoost ensures high accuracy and robustness.

## Author
**Ritesh Mishra**

## License
This project is open-source and available under the MIT License.

