# Configurations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import product

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, PrecisionRecallDisplay, f1_score
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import pickle
import os

from preprocessing import preprocess_data, split_dataset, scale_datasets, encode_with_dv
from model_wrapper import StartupFailureModel

print("Current working directory:", os.getcwd())
print("Files in this folder:", os.listdir())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Parameters

C_final = 0.1
penalty_final = 'l2'
max_iter_final = 10000
random_state_final = 42
output_file = f'model_C={C_final}.bin'

print("\n================================================")
print("Starting Machine Learning model training...")
print("================================================\n")
print("The selected parameters are:")
print("  - Model: Logistic Regression")
print(f"  - Parameter C: {C_final}")
print(f"  - Penalty: {penalty_final}\n")


# Data Loading

csv_path = os.path.join(os.path.dirname(__file__), 'startup_failure_prediction.csv')
df = pd.read_csv(csv_path, encoding='ISO-8859-1')


# Data Pre-Processing

df_processed, categorical, numerical = preprocess_data(df)

df_splits, y_splits = split_dataset(df_processed, target_col='failed')

scaled_splits, scaler = scale_datasets(df_splits, numerical)

X_splits, dv = encode_with_dv(scaled_splits, categorical, numerical)

X_train      = X_splits['train']
X_val        = X_splits['val']
X_test       = X_splits['test']
X_full_train = X_splits['full_train']

y_train = y_splits['train']
y_val   = y_splits['val']
y_test  = y_splits['test']
y_full_train = y_splits['full_train']


# Training the Final Model

final_model = LogisticRegression(
    C=C_final,
    penalty=penalty_final,
    max_iter=max_iter_final,
    random_state=random_state_final,
    class_weight='balanced'
)

final_model.fit(X_full_train, y_full_train)

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

acc_test = accuracy_score(y_test, y_test_pred)
auc_test = roc_auc_score(y_test, y_test_proba)


# Creating Wrapper

model_wrapper = StartupFailureModel(
    dv=dv,
    scaler=scaler,
    model=final_model,
    categorical=categorical,
    numerical=numerical
)


# Saving the Model

with open(output_file, 'wb') as f_out:
    pickle.dump(model_wrapper, f_out)

print(f"✅ Model successfully saved as {output_file}")

print("\n📊 Test set performance:")
print(f"- Accuracy: {acc_test:.4f}")
print(f"- AUC: {auc_test:.4f}")
