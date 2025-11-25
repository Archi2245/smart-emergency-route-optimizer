# src/ml_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(df, epochs=25, batch_size=32):
    """
    Train ANN model to predict congestion levels.
    
    Architecture:
    - Input: 6 features (hour, day, holiday, distance, base_speed, base_congestion)
    - Hidden Layer 1: 64 neurons, ReLU activation, Dropout(0.3)
    - Hidden Layer 2: 32 neurons, ReLU activation, Dropout(0.2)
    - Output Layer: 3 neurons (softmax) for 3 congestion classes
    
    Returns:
        model: trained Keras model
        scaler: fitted StandardScaler
    """
    print("=== Training ANN Congestion Predictor ===")
    
    # Prepare features and labels
    feature_cols = ['hour', 'day_of_week', 'is_holiday', 
                    'distance_km', 'base_speed_kmh', 'base_congestion']
    X = df[feature_cols].values
    y = df['congestion_level'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Build ANN model
    model = keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(64, activation='relu', name='hidden1'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', name='hidden2'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_test)} samples...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Training Complete!")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model and scaler
    os.makedirs('data', exist_ok=True)
    model.save('data/congestion_model.keras')
    joblib.dump(scaler, 'data/scaler.pkl')
    print("Model saved to data/congestion_model.keras")
    
    return model, scaler


def load_model_and_scaler():
    """Load pre-trained model and scaler."""
    model = keras.models.load_model('data/congestion_model.keras')
    scaler = joblib.load('data/scaler.pkl')
    print("Model and scaler loaded successfully")
    return model, scaler


def predict_congestion_keras(model, scaler, edge_meta, hour, day_of_week, is_holiday):
    """
    Predict congestion level for a specific edge at given time.
    
    Returns:
        predicted_level: int (0=Low, 1=Medium, 2=High)
        probabilities: array of probabilities for each class
    """
    # Prepare input
    features = np.array([[
        hour,
        day_of_week,
        is_holiday,
        edge_meta['distance_km'],
        edge_meta['base_speed_kmh'],
        edge_meta['base_congestion']
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    probs = model.predict(features_scaled, verbose=0)[0]
    predicted_level = int(np.argmax(probs))
    
    return predicted_level, probs


def estimate_travel_time(distance_km, base_speed_kmh, congestion_level):
    """
    Estimate travel time based on congestion.
    
    Congestion factors:
    - Low (0): 1.0x (normal speed)
    - Medium (1): 0.6x (40% slower)
    - High (2): 0.3x (70% slower)
    """
    congestion_factors = {0: 1.0, 1: 0.6, 2: 0.3}
    factor = congestion_factors.get(congestion_level, 1.0)
    
    effective_speed = base_speed_kmh * factor
    time_hours = distance_km / effective_speed
    time_minutes = time_hours * 60
    
    return time_minutes