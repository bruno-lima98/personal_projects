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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Wrapper

class StartupFailureModel:
    def __init__(self, dv, scaler, model, categorical, numerical):
        self.dv = dv
        self.scaler = scaler
        self.model = model
        self.categorical = categorical
        self.numerical = numerical

    def preprocess(self, df):
        df_copy = df.copy()
        df_copy[self.numerical] = self.scaler.transform(df_copy[self.numerical])
        dicts = df_copy[self.categorical + self.numerical].to_dict(orient='records')
        X = self.dv.transform(dicts)
        return X

    def predict(self, df):
        X = self.preprocess(df)
        return self.model.predict(X)

    def predict_proba(self, df):
        X = self.preprocess(df)
        return self.model.predict_proba(X)[:, 1]

# Parameters

C_final = 0.1
penalty_final = 'l2'
max_iter_final = 10000
random_state_final = 42
output_file = f'model_C={C_final}.bin'


# Data Preparation

df = pd.read_csv('startup_failure_prediction.csv', encoding='ISO-8859-1')

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
drop = ['permalink', 'name', 'homepage_url', 'category_list',
        'founded_month', 'founded_quarter', 'founded_year']

for col in drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df = df[df['status'].isin(['acquired', 'closed'])].copy()
df['failed'] = df['status'].map({'closed': 1, 'acquired': 0})
df.drop('status', axis=1, inplace=True)
df['failed'].value_counts()


df_clean = df.copy()

df_clean['founded_at'] = pd.to_datetime(df_clean['founded_at'], errors='coerce')
df_clean['first_funding_at'] = pd.to_datetime(df_clean['first_funding_at'], errors='coerce')
df_clean['last_funding_at'] = pd.to_datetime(df_clean['last_funding_at'], errors='coerce')

df_clean['funding_total_usd'] = df_clean['funding_total_usd'].str.strip().str.lower().str.replace(',', '')
df_clean['funding_total_usd'] = pd.to_numeric(df_clean['funding_total_usd'], errors='coerce')

df_clean.loc[df_clean['country_code'] != 'USA', 'state_code'] = 'FGN'
df_clean.loc[df_clean['country_code'] != 'USA', 'country_code'] = 'FGN'

drop = ['region', 'city']

for col in drop:
    if col in df_clean.columns:
        df_clean.drop(columns=col, inplace=True)

null_cols = df_clean.isnull().sum()
null_cols = null_cols[null_cols > 0]

df_clean = df_clean.dropna(subset=['first_funding_at'])


df_clean['market'] = df_clean['market'].fillna('Unknown')

df_clean['founded_missing'] = df_clean['founded_at'].isna().astype(int)

df_clean['founded_year'] = df_clean['founded_at'].dt.year
df_clean['founded_month'] = df_clean['founded_at'].dt.month

df_clean['first_funding_year'] = df_clean['first_funding_at'].dt.year
df_clean['first_funding_month'] = df_clean['first_funding_at'].dt.month

df_clean['last_funding_year'] = df_clean['last_funding_at'].dt.year
df_clean['last_funding_month'] = df_clean['last_funding_at'].dt.month

df_clean['founded_year'] = df_clean['founded_year'].fillna(df_clean['founded_year'].median())
df_clean['founded_month'] = df_clean['founded_month'].fillna(df_clean['founded_month'].mode()[0])

df_clean['funding_total_usd'] = df_clean['funding_total_usd'].fillna(0)

df_clean['months_to_first_funding'] = ((df_clean['first_funding_at'] - pd.to_datetime(df_clean['founded_year'].astype(int).astype(str) 
                                                                                      + '-' 
                                                                                      + df_clean['founded_month'].astype(int).astype(str) + '-01')) 
                                       / pd.Timedelta(days=30)).round(1)

df_clean['funding_duration_months'] = ((df_clean['last_funding_at'] - df_clean['first_funding_at']) / pd.Timedelta(days=30)).round(1)

df_clean.drop(['first_funding_at', 'last_funding_at', 'founded_at'], axis=1, inplace=True)

categorical = []
numerical = []

for col in df_clean.columns:
    if col == 'failed':
        continue
    if df_clean[col].dtype == 'object':
        categorical.append(col)
    else:
        numerical.append(col)

df_clean = df_clean[df_clean['founded_year'] >= 2000]
df_final = df_clean.copy()

df_final = df_final[
    (df_final['first_funding_year'] >= df_final['founded_year']) &
    (df_final['last_funding_year'] >= df_final['first_funding_year']) &
    (df_final['months_to_first_funding'] >= 0)
]

for col in categorical:
    df_final[col] = df_final[col].str.strip().str.lower().str.replace(' ', '_')


market_counts = df_final['market'].value_counts()

market_perc = market_counts / market_counts.sum() * 100

market_summary = pd.DataFrame({
    'count': market_counts,
    'percent': market_perc,
    'cumulative_percent': market_perc.cumsum()
})

top_markets = df_final['market'].value_counts().head(25).index
df_final['market'] = df_final['market'].apply(lambda x: x if x in top_markets else 'other')


state_counts = df_final['state_code'].value_counts()

state_perc = state_counts / state_counts.sum() * 100

state_summary = pd.DataFrame({
    'count': state_counts,
    'percent': state_perc,
    'cumulative_percent': state_perc.cumsum()
})

top_states = df_final['state_code'].value_counts().head(12).index
df_final['state_code'] = df_final['state_code'].apply(lambda x: x if x in top_states else 'other')

for col in categorical:
    print(f'Feature: {col}.')
    print(f'- Number of Categories: {df_final[col].nunique()}.')

df_final[numerical].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).T

investment_cols = [
    'seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
    'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
    'post_ipo_debt', 'secondary_market', 'product_crowdfunding'
]

for col in investment_cols:
    df_final[f'received_{col}'] = (df_final[col] > 0).astype(int)

df_final.drop(columns=investment_cols, inplace=True)

round_cols = ['round_a','round_b','round_c','round_d','round_e','round_f','round_g','round_h']

df_final.drop(columns=round_cols, inplace=True)

numerical_att = [
    'funding_total_usd', 'funding_rounds', 'founded_year', 'founded_month', 'first_funding_year', 'first_funding_month', 
    'last_funding_year', 'last_funding_month', 'months_to_first_funding', 'funding_duration_months', 
    ]
flags = [
    'received_seed', 'received_venture', 'received_equity_crowdfunding', 'received_undisclosed', 'received_convertible_note', 
    'received_debt_financing', 'received_angel', 'received_grant', 'received_private_equity', 'received_post_ipo_equity', 
    'received_post_ipo_debt', 'received_secondary_market', 'received_product_crowdfunding', 'founded_missing'
    ]

df_final[numerical_att].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).T

log_treatment = ['funding_total_usd', 'months_to_first_funding', 'funding_duration_months']

for col in log_treatment:
    df_final[col] = np.log1p(df_final[col])

proportions = df_final['failed'].value_counts(normalize=True)

numerical_comp = [
    'funding_total_usd',
    'funding_rounds',
    'months_to_first_funding',
    'funding_duration_months'
]

def split_dataset(df, target_col='failed', test_size=0.2, val_size=0.25, random_state=1):

    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val       = train_test_split(df_full_train, test_size=val_size, random_state=random_state)

    for d in [df_full_train, df_train, df_val, df_test]:
        d.reset_index(drop=True, inplace=True)

    y_full_train = df_full_train[target_col].values
    y_train      = df_train[target_col].values
    y_val        = df_val[target_col].values
    y_test       = df_test[target_col].values

    for d in [df_full_train, df_train, df_val, df_test]:
        d.drop(columns=[target_col], inplace=True)

    df_splits = {
        'full_train': df_full_train,
        'train': df_train,
        'val': df_val,
        'test': df_test
    }

    y_splits = {
        'full_train': y_full_train,
        'train': y_train,
        'val': y_val,
        'test': y_test
    }

    return df_splits, y_splits

df_splits, y_splits = split_dataset(df_final)

df_full_train = df_splits['full_train']
df_train      = df_splits['train']
df_val        = df_splits['val']
df_test       = df_splits['test']

y_full_train  = y_splits['full_train']
y_train       = y_splits['train']
y_val         = y_splits['val']
y_test        = y_splits['test']

def scale_datasets(datasets, cols):
    scaler = StandardScaler()
    scaled_sets = {}

    scaler.fit(datasets['train'][cols])

    for name, df in datasets.items():
        scaled_sets[name] = df.copy()
        scaled_sets[name][cols] = scaler.transform(df[cols])

    return scaled_sets, scaler

dfs = {
    'train': df_train,
    'val': df_val,
    'test': df_test,
    'full_train': df_full_train
}

scaled, scaler = scale_datasets(dfs, numerical_att)

df_train_scaled      = scaled['train']
df_val_scaled        = scaled['val']
df_test_scaled       = scaled['test']
df_full_train_scaled = scaled['full_train']

def encode_with_dv(df_splits, categorical, numerical):
    cols = categorical + numerical

    dicts = {name: df[cols].to_dict(orient='records') for name, df in df_splits.items()}

    dv = DictVectorizer(sparse=False)
    dv.fit(dicts['train'])

    X_splits = {name: dv.transform(dicts[name]) for name in dicts}

    return X_splits, dv

X_splits, dv = encode_with_dv(df_splits, categorical, numerical_att)

X_train      = X_splits['train']
X_val        = X_splits['val']
X_test       = X_splits['test']
X_full_train = X_splits['full_train']


# Training the Model

def train_and_evaluate(X_train, y_train, X_val, y_val, C=1.0, max_iter=1000, random_state=42, class_weight="balanced", label="Validation"):

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    cm = confusion_matrix(y_val, y_pred)

    return model, {"accuracy": acc, "auc": auc, "confusion_matrix": cm}


model, metrics = train_and_evaluate(
    X_train, y_train, X_val, y_val,
    C=1.0,
    max_iter=10000,
    random_state=42,
    class_weight='balanced',
    label="Validation"
)


# Validating the Model

def run_kfold(df, target_col, categorical, numerical, n_splits=5, C=1.0, max_iter=1000, random_state=42):

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    fold = 1
    for train_idx, val_idx in kfold.split(df):

        df_train = df.iloc[train_idx].copy()
        df_val   = df.iloc[val_idx].copy()

        y_train = df_train[target_col].values
        y_val   = df_val[target_col].values

        df_train = df_train.drop(columns=[target_col])
        df_val   = df_val.drop(columns=[target_col])

        df_splits = {'train': df_train, 'val': df_val}
        scaled_sets = scale_datasets(df_splits, numerical)
        X_splits, dv = encode_with_dv(scaled_sets, categorical, numerical)

        model, metrics = train_and_evaluate(
            X_splits['train'],
            y_train,
            X_splits['val'],
            y_val,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            label=f"Fold {fold}"
        )

        scores.append(metrics["auc"])
        fold += 1

run_kfold(
    df=df_final,
    target_col='failed',
    categorical=categorical,
    numerical=numerical_att,
    n_splits=5,
    C=1.0,
    max_iter=10000
)

C_values = [0.01, 0.1, 1, 10]
penalties = ['l1', 'l2']
results = []

for C, penalty in product(C_values, penalties):

    try:
        model, metrics = train_and_evaluate(
            X_train, y_train,
            X_val, y_val,
            C=C,
            max_iter=10000,
            random_state=42,
            class_weight='balanced',
            label=f"Validation (C={C}, penalty={penalty})"
        )

        results.append({
            'C': C,
            'penalty': penalty,
            'AUC': metrics['auc'],
            'Accuracy': metrics['accuracy']
        })

    except Exception as e:
        print(f"⚠️ Error with C={C}, penalty={penalty}: {e}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='AUC', ascending=False).reset_index(drop=True)


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
cm_test = confusion_matrix(y_test, y_test_pred)


# Creating Wrapper

model_wrapper = StartupFailureModel(
    dv=dv,
    scaler=scaler,
    model=final_model,
    categorical=categorical,
    numerical=numerical_att
)


# Saving the Model

with open(output_file, 'wb') as f_out:
    pickle.dump(model_wrapper, f_out)

print(f"✅ Modelo salvo com sucesso em {output_file}")