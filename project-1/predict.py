import pickle
import pandas as pd

with open('model_C=0.1.bin', 'rb') as f_in:
    model_wrapper = pickle.load(f_in)

startup = pd.DataFrame({
    'funding_total_usd': [1000000],
    'funding_rounds': [2],
    'founded_year': [2018],
    'founded_month': [5],
    'first_funding_year': [2018],
    'first_funding_month': [6],
    'last_funding_year': [2019],
    'last_funding_month': [3],
    'months_to_first_funding': [1.1],
    'funding_duration_months': [21.0],
    'received_seed': [1],
    'received_venture': [0],
    'received_equity_crowdfunding': [0],
    'received_undisclosed': [0],
    'received_convertible_note': [0],
    'received_debt_financing': [0],
    'received_angel': [0],
    'received_grant': [0],
    'received_private_equity': [0],
    'received_post_ipo_equity': [0],
    'received_post_ipo_debt': [0],
    'received_secondary_market': [0],
    'received_product_crowdfunding': [0],
    'founded_missing': [0],
    'market': ['software'],
    'state_code': ['ca']
})

pred_class = model_wrapper.predict(startup)
pred_proba = model_wrapper.predict_proba(startup)

print("input ", startup)
print("Predict Class: ", pred_class)
print("Probability of Failure: ", pred_proba)
