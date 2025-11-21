# backend/src/ml_model.py
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

SCALER_PATH = "data/scaler.pkl"
MODEL_PATH = "data/ann_congestion_model.h5"

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(df_hist, features=['hour','day','is_holiday','base_speed','distance','base_congestion'], epochs=25):
    X = df_hist[features].values
    y = df_hist['congestion_level'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, verbose=2)
    os.makedirs('data', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    model.save(MODEL_PATH)
    return model, scaler

def load_model_and_scaler():
    import joblib
    from tensorflow.keras.models import load_model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Train model first.")
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
    return model, scaler

def predict_congestion_keras(model, scaler, edge_meta, hour, day, is_holiday):
    feat = np.array([[hour, day, int(is_holiday), edge_meta['base_speed_kmph'], edge_meta['distance_km'], edge_meta['base_congestion']]])
    Xs = scaler.transform(feat)
    probs = model.predict(Xs, verbose=0)[0]
    return int(np.argmax(probs)), probs
