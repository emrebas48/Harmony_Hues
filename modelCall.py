#DENEME KODU
# libraries
import pandas as pd

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

path = 'data/spotify_data.csv'
df = pd.read_csv(path)

df.dropna(inplace=True)

print(df)

print(df.dtypes)

# train and test
# Create DataFrame X by dropping specified columns
X = df.drop(columns=["artist_name", "track_name", "track_id", "genre"])

# Load the saved model from the file
loaded_model1 = joblib.load('trained_model_energy.pkl')
loaded_model2 = joblib.load('trained_model_liveness.pkl')
loaded_model3 = joblib.load('trained_model_tempo.pkl')
# New data for prediction (replace this with your new data)
new_data = X.iloc[[1000]]

print(new_data)

# Use the loaded model to make predictions on the new data
prediction1 = loaded_model1.predict(new_data)
prediction2 = loaded_model2.predict(new_data)
prediction3 = loaded_model3.predict(new_data)
# Display the predictions
print("Prediction1:", prediction1)
print("Prediction2:", prediction2)
print("Prediction3:", prediction3)