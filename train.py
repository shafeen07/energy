import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

os.chdir(r'C:\Users\shafe\OneDrive\Desktop\Projects')

df = pd.read_csv('energy.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(40, activation='relu', input_shape=(X_train.shape[1], )),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, batch_size=20, validation_split=0.1)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")
model.save('energy_model.h5')
import joblib
joblib.dump(scaler, 'scaler.save')
print("Model and scaler saved")