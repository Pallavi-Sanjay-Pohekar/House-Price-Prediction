import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Simple dataset with  made-up values
data = {
    'bedrooms': [2, 3, 4, 3, 5, 2, 4, 3, 5, 6],
    'bathrooms': [1, 2, 2, 3, 3, 1, 2, 2, 4, 4],
    'size': [1000, 1500, 2000, 1800, 2500, 1100, 2300, 1600, 2700, 3000],
    'location': [1, 2, 3, 1, 2, 3, 1, 3, 2, 1],  # Consider location as numbers
    'price': [150000, 200000, 250000, 230000, 300000, 170000, 280000, 220000, 350000, 400000]
}

df = pd.DataFrame(data)

# Split dataset
X = df[['bedrooms', 'bathrooms', 'size', 'location']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
