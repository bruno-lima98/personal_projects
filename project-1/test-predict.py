import requests

url = 'http://localhost:9696/predict'

startup = {
    "permalink": "/organization/waywire",
    "name": "#waywire",
    "homepage_url": "http://www.waywire.com",
    "category_list": "|Entertainment|Politics|Social Media|News|",
    "market": "News",
    "funding_total_usd": " 17,50,000 ",
    "status": "acquired",
    "country_code": "USA",
    "state_code": "NY",
    "region": "New York",
    "city": "New York",
    "funding_rounds": 1,
    "founded_at": "2012-06-01",
    "founded_month": "2012-06",
    "founded_quarter": "2012-Q2",
    "founded_year": 2012,
    "first_funding_at": "2012-06-30",
    "last_funding_at": "2012-06-30",
    "seed": 1750000,
    "venture": 0,
    "equity_crowdfunding": 0,
    "undisclosed": 0,
    "convertible_note": 0,
    "debt_financing": 0,
    "angel": 0,
    "grant": 0,
    "private_equity": 0,
    "post_ipo_equity": 0,
    "post_ipo_debt": 0,
    "secondary_market": 0,
    "product_crowdfunding": 0,
    "round_A": 0,
    "round_B": 0,
    "round_C": 0,
    "round_D": 0,
    "round_E": 0,
    "round_F": 0,
    "round_G": 0,
    "round_H": 0
}

response = requests.post(url, json=startup).json()

print(response)

if response['failure'] == True:
    print('This startup probably will fail. Be careful.')
else:
    print('This startup seems to have a good future!')