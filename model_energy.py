# libraries
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
#from sklearn.externals import joblib  # For scikit-learn <= 0.23
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    path = 'data/spotify_data.csv'
    df = pd.read_csv(path)

    df.dropna(inplace=True)

    print(df)

    print(df.dtypes)

    # train and test
    # Create DataFrame X by dropping specified columns
    X = df.drop(columns=["artist_name", "track_name", "track_id", "genre"])

    # Create a copy of 'Energy' column from df to a new DataFrame Y
    #Y = pd.DataFrame(df["energy", "liveness", "tempo"])

    columns_to_keep = ['energy']
    Y = df[columns_to_keep].copy()

    # Min-Max Scaling Energy
    scaled_min = 0
    scaled_max = 255

    min_value = df['energy'].min()
    max_value = df['energy'].max()

    # Scale values between 0 and 1
    Y['energy_scaled'] = (df['energy'] - min_value) / (max_value - min_value)

    # Scale to a range between 0 and 255

    Y['energy_scaled'] = Y['energy_scaled'] * (scaled_max - scaled_min) + scaled_min

    Y = Y.drop(columns=["energy"])

    Y = Y.rename(columns={'energy_scaled': 'energy'})
    Y[['energy']] = Y[['energy']].astype(int)

    '''# Min-Max Scaling - liveness


    min_value = df['liveness'].min()
    max_value = df['liveness'].max()

    # Scale values between 0 and 1
    Y['liveness_scaled'] = (df['liveness'] - min_value) / (max_value - min_value)

    # Scale to a range between 0 and 255
    Y['liveness_scaled'] = Y['liveness_scaled'] * (scaled_max - scaled_min) + scaled_min

    Y = Y.drop(columns=["liveness"])

    Y = Y.rename(columns={'liveness_scaled': 'liveness'})
    Y[['liveness']] = Y[['liveness']].astype(int)

    # Min-Max Scaling - tempo
    min_value = df['tempo'].min()
    max_value = df['tempo'].max()

    # Scale values between 0 and 1
    Y['tempo_scaled'] = (df['tempo'] - min_value) / (max_value - min_value)

    # Scale to a range between 0 and 255

    Y['tempo_scaled'] = Y['tempo_scaled'] * (scaled_max - scaled_min) + scaled_min

    Y = Y.drop(columns=["tempo"])

    Y = Y.rename(columns={'tempo_scaled': 'tempo'})
    Y[['tempo']] = Y[['tempo']].astype(int)'''


    # Use ravel() to convert y to a 1D array
    Y = Y.values.ravel()

    print(Y)



    # Splitting the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Initializing and fitting a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")



    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initializing and fitting a Ridge Regression model
    ridge_model = Ridge(alpha=0.5)  # Adjust alpha as needed
    ridge_model.fit(X_train_scaled, y_train)

    # Making predictions on the scaled test set
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    # Evaluating the Ridge Regression model
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print(f"Ridge Mean Squared Error (MSE): {mse_ridge}")
    print(f"Ridge R-squared (R2): {r2_ridge}")

    # Save the trained model to a file
    joblib.dump(model, 'trained_model_energy.pkl')



