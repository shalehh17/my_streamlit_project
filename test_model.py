import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

print("1. Membaca data train.csv...")
# Pastikan train.csv ada di folder data/
df = pd.read_csv("data/train.csv")

print("2. Menyiapkan fitur dan target...")
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "1stFlrSF"]
target = "SalePrice"

X = df[features].fillna(0)
y = df[target]

# Pastikan folder models/ ada
if not os.path.exists("models"):
    os.makedirs("models")

print("3. Melatih dan menyimpan Model 1: Linear Regression...")
linreg = LinearRegression()
linreg.fit(X, y)
with open("models/linreg_model.pkl", "wb") as f:
    pickle.dump(linreg, f)

print("4. Melatih dan menyimpan Model 2: Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("5. Melatih dan menyimpan Model 3: XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X, y)
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("\n✅ SUKSES! Ketiga model berhasil dilatih. Silakan cek folder 'models/'.")