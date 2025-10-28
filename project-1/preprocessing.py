import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# =========================================================================================================
# ===================================== 1 - PREPROCESSING THE DATASET =====================================
# =========================================================================================================

def preprocess_data(df):
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    drop = ['permalink', 'name', 'homepage_url', 'category_list',
        'founded_month', 'founded_quarter', 'founded_year']

    for col in drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    df = df[df['status'].isin(['acquired', 'closed'])].copy()
    df['failed'] = df['status'].map({'closed': 1, 'acquired': 0})
    df.drop('status', axis=1, inplace=True)
    
    df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
    df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
    df['last_funding_at'] = pd.to_datetime(df['last_funding_at'], errors='coerce')
    
    df['funding_total_usd'] = df['funding_total_usd'].str.strip().str.lower().str.replace(',', '')
    df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
    
    df.loc[df['country_code'] != 'USA', 'state_code'] = 'FGN'
    df.loc[df['country_code'] != 'USA', 'country_code'] = 'FGN'
    
    drop_cols = ['region', 'city']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    df = df.dropna(subset=['first_funding_at'])
    
    df['market'] = df['market'].fillna('Unknown')
    
    df['founded_missing'] = df['founded_at'].isna().astype(int)
    
    df['founded_year'] = df['founded_at'].dt.year
    df['founded_month'] = df['founded_at'].dt.month
    df['first_funding_year'] = df['first_funding_at'].dt.year
    df['first_funding_month'] = df['first_funding_at'].dt.month
    df['last_funding_year'] = df['last_funding_at'].dt.year
    df['last_funding_month'] = df['last_funding_at'].dt.month
    
    df['founded_year'] = df['founded_year'].fillna(df['founded_year'].median())
    df['founded_month'] = df['founded_month'].fillna(df['founded_month'].mode()[0])
    df['funding_total_usd'] = df['funding_total_usd'].fillna(0)
    
    df['months_to_first_funding'] = ((df['first_funding_at'] - 
                                      pd.to_datetime(df['founded_year'].astype(int).astype(str) + '-' +
                                                     df['founded_month'].astype(int).astype(str) + '-01')) 
                                     / pd.Timedelta(days=30)).round(1)
    
    df['funding_duration_months'] = ((df['last_funding_at'] - df['first_funding_at']) / pd.Timedelta(days=30)).round(1)
    
    df.drop(columns=['first_funding_at', 'last_funding_at', 'founded_at'], inplace=True)
    
    df = df[df['founded_year'] >= 2000]
    df = df[(df['first_funding_year'] >= df['founded_year']) &
            (df['last_funding_year'] >= df['first_funding_year']) &
            (df['months_to_first_funding'] >= 0)]
    
    categorical = []
    numerical = []
    for col in df.columns:
        if col == 'failed':
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower().str.replace(' ', '_')
            categorical.append(col)
        else:
            numerical.append(col)
    
    top_markets = df['market'].value_counts().head(25).index
    df['market'] = df['market'].apply(lambda x: x if x in top_markets else 'other')
    
    top_states = df['state_code'].value_counts().head(12).index
    df['state_code'] = df['state_code'].apply(lambda x: x if x in top_states else 'other')

    investment_cols = [
        'seed', 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
        'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
        'post_ipo_debt', 'secondary_market', 'product_crowdfunding'
    ]
    for col in investment_cols:
        if col in df.columns:
            df[f'received_{col}'] = (df[col] > 0).astype(int)
    df.drop(columns=investment_cols, inplace=True)
    
    round_cols = ['round_a','round_b','round_c','round_d','round_e','round_f','round_g','round_h']
    df.drop(columns=round_cols, inplace=True)
    
    numerical_att = [
        'funding_total_usd', 'funding_rounds', 'founded_year', 'founded_month',
        'first_funding_year', 'first_funding_month', 'last_funding_year', 'last_funding_month',
        'months_to_first_funding', 'funding_duration_months'
    ]
    
    flags = [
        'received_seed', 'received_venture', 'received_equity_crowdfunding', 'received_undisclosed', 'received_convertible_note', 
        'received_debt_financing', 'received_angel', 'received_grant', 'received_private_equity', 'received_post_ipo_equity', 
        'received_post_ipo_debt', 'received_secondary_market', 'received_product_crowdfunding', 'founded_missing'
    ]
    
    log_treatment = ['funding_total_usd', 'months_to_first_funding', 'funding_duration_months']
    for col in log_treatment:
        df[col] = np.log1p(df[col])
    
    numerical_att = numerical_att
    categorical = categorical
    
    return df, categorical, numerical_att

# =========================================================================================================
# ======================================== 2 - SPLITING THE DATASET =======================================
# =========================================================================================================

def split_dataset(df, target_col='failed', test_size=0.2, val_size=0.25, random_state=1):
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    
    for d in [df_full_train, df_train, df_val, df_test]:
        d.reset_index(drop=True, inplace=True)
    
    y_full_train = df_full_train[target_col].values
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values
    
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

# =========================================================================================================
# ================================== 3 - NORMALIZATION AND STANDARTOZATION ================================
# =========================================================================================================

def scale_datasets(datasets, cols):
    scaler = StandardScaler()
    scaled_sets = {}
    scaler.fit(datasets['train'][cols])
    
    for name, df in datasets.items():
        scaled_sets[name] = df.copy()
        scaled_sets[name][cols] = scaler.transform(df[cols])
    
    return scaled_sets, scaler

# =========================================================================================================
# ========================================== 4 - ONE-HOT ENCODING =========================================
# =========================================================================================================

def encode_with_dv(df_splits, categorical, numerical):
    cols = categorical + numerical
    dicts = {name: df[cols].to_dict(orient='records') for name, df in df_splits.items()}
    dv = DictVectorizer(sparse=False)
    dv.fit(dicts['train'])
    X_splits = {name: dv.transform(dicts[name]) for name in dicts}
    return X_splits, dv