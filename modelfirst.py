import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data_for_lstm(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for LSTM model training

    Args:
    X (numpy.ndarray): Input features
    y (numpy.ndarray): Target values
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility

    Returns:
    Tuple of scaled training and testing datasets
    """
    # Reshape X for LSTM input (samples, time steps, features)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the target variable
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train_scaled, y_test_scaled, target_scaler


def create_lstm_model(input_shape, units=50, dropout=0.2):
    """
    Create an LSTM model for time series prediction

    Args:
    input_shape (tuple): Shape of input data
    units (int): Number of LSTM units
    dropout (float): Dropout rate

    Returns:
    Keras Sequential model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout),

        # Second LSTM layer
        LSTM(units // 2),
        Dropout(dropout),

        # Dense output layer
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model


def train_lstm_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the LSTM model

    Args:
    X_train, y_train, X_test, y_test: Training and testing data
    epochs (int): Number of training epochs
    batch_size (int): Batch size for training

    Returns:
    Trained model and training history
    """
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create the model
    model = create_lstm_model(input_shape)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test_scaled, target_scaler):
    """
    Evaluate the model performance

    Args:
    model: Trained Keras model
    X_test: Test features
    y_test_scaled: Scaled test targets
    target_scaler: Scaler used for target variable

    Returns:
    Dictionary of performance metrics
    """
    # Make predictions
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test = target_scaler.inverse_transform(y_test_scaled)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }


def plot_training_history(history):
    """
    Plot training and validation loss

    Args:
    history: Keras training history
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/processed/training_history.png')
    plt.close()


def main():
    # Load preprocessed data
    X = np.load('data/processed/X_features.npy')
    y = np.load('data/processed/y_targets.npy')

    # Prepare data for LSTM
    X_train, X_test, y_train_scaled, y_test_scaled, target_scaler = prepare_data_for_lstm(X, y)

    # Train the model
    model, history = train_lstm_model(X_train, y_train_scaled, X_test, y_test_scaled)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test_scaled, target_scaler)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('data/processed/tmax_lstm_model.h5')
    print("Model saved to data/processed/tmax_lstm_model.h5")


if __name__ == "__main__":
    main()