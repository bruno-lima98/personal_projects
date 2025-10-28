from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

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
