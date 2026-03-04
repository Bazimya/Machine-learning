import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample dataset
data = {
    "size": [120, 200, 150, 300, 250, 180, 220],
    "bedrooms": [3, 4, 3, 5, 4, 3, 4],
    "bathrooms": [2, 3, 2, 4, 3, 2, 3],
    "location_score": [7, 9, 6, 10, 8, 7, 9],
    "price": [50, 90, 55, 150, 120, 70, 110]
}

df = pd.DataFrame(data)

X = df.drop("price", axis=1)
y = df["price"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully!")