import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def create_lstm_model():
    """
    Create and train LSTM model with proper error handling
    """
    try:
        # Try importing TensorFlow
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        # Suppress TensorFlow warnings
        tf.get_logger().setLevel("ERROR")

        print("TensorFlow imported successfully")
        print(f"TensorFlow version: {tf.__version__}")

    except ImportError as e:
        print(f"TensorFlow import error: {e}")
        print("Please install TensorFlow: pip install tensorflow")
        return None
    except Exception as e:
        print(f"TensorFlow initialization error: {e}")
        print("Falling back to alternative approach...")
        return None

    try:
        # Load data
        print("Loading data...")
        data = pd.read_csv("stock_data.csv")

        # Check if required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume", "Target"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            print(f"Available columns: {data.columns.tolist()}")
            return None

        print(f"Data loaded successfully. Shape: {data.shape}")

        # Prepare features
        features = ["Open", "High", "Low", "Close", "Volume"]
        X = data[features].values
        y = data["Target"].values

        # Scale features
        print("Scaling features...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Create sequences for LSTM
        seq_len = 60
        if len(X_scaled) < seq_len + 1:
            print(
                f"Not enough data for sequence length {seq_len}. Need at least {seq_len + 1} samples."
            )
            return None

        Xs, ys = [], []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len : i])
            ys.append(y[i])

        Xs, ys = np.array(Xs), np.array(ys)
        print(f"Created sequences. X shape: {Xs.shape}, y shape: {ys.shape}")

        # Split data
        split_idx = int(0.8 * len(Xs))
        X_train, X_test = Xs[:split_idx], Xs[split_idx:]
        y_train, y_test = ys[:split_idx], ys[split_idx:]

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        # Build LSTM model
        print("Building LSTM model...")
        model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(seq_len, 5)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation="sigmoid"),
            ]
        )

        # Compile model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        print("Model architecture:")
        model.summary()

        # Train model
        print("Training model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            shuffle=False,  # Important for time series data
        )

        # Evaluate model
        print("Evaluating model...")
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")

        # Save model
        try:
            model.save("lstm_model.h5")
            print("Model saved as 'lstm_model.h5'")
        except Exception as e:
            print(f"Error saving model: {e}")

        # Save scaler for future use
        try:
            import joblib

            joblib.dump(scaler, "lstm_scaler.pkl")
            print("Scaler saved as 'lstm_scaler.pkl'")
        except Exception as e:
            print(f"Error saving scaler: {e}")

        return model, scaler, history

    except Exception as e:
        print(f"Error in LSTM model creation: {e}")
        import traceback

        traceback.print_exc()
        return None


def predict_with_lstm(
    model_path="lstm_model.h5",
    scaler_path="lstm_scaler.pkl",
    data_path="stock_data.csv",
):
    """
    Load trained model and make predictions
    """
    try:
        import joblib
        import tensorflow as tf

        # Load model and scaler
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Load recent data for prediction
        data = pd.read_csv(data_path)
        features = ["Open", "High", "Low", "Close", "Volume"]
        X = data[features].values[-60:]  # Last 60 days

        # Scale and reshape for prediction
        X_scaled = scaler.transform(X)
        X_pred = X_scaled.reshape(1, 60, 5)

        # Make prediction
        prediction = model.predict(X_pred)
        prob = prediction[0][0]

        print(f"Prediction probability: {prob:.4f}")
        print(f"Predicted direction: {'UP' if prob > 0.5 else 'DOWN'}")

        return prob

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


if __name__ == "__main__":
    print("Starting LSTM model training...")
    result = create_lstm_model()

    if result is not None:
        print("LSTM model training completed successfully!")

        # Test prediction
        print("\nTesting prediction...")
        try:
            predict_with_lstm()
        except Exception as e:
            print(f"Prediction test failed: {e}")
    else:
        print("LSTM model training failed!")
        print("\nTrying alternative approaches...")

        # Alternative: Simple feedforward neural network
        try:
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.model_selection import train_test_split
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler

            print("Using scikit-learn MLPClassifier as alternative...")

            data = pd.read_csv("stock_data.csv")
            features = ["Open", "High", "Low", "Close", "Volume"]
            X = data[features].values
            y = data["Target"].values

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create and train model
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            )

            mlp.fit(X_train, y_train)

            # Evaluate
            train_pred = mlp.predict(X_train)
            test_pred = mlp.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            print(f"MLP Training Accuracy: {train_acc:.4f}")
            print(f"MLP Testing Accuracy: {test_acc:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, test_pred))

            # Save the alternative model
            import joblib

            joblib.dump(mlp, "mlp_model.pkl")
            joblib.dump(scaler, "mlp_scaler.pkl")
            print("Alternative MLP model saved!")

        except Exception as e:
            print(f"Alternative model also failed: {e}")
            print("Please check your data and dependencies.")
