import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


def create_sequences(X, y, seq_len=10):
    """
    Create sequences for time series prediction using a smaller window
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        # Flatten the sequence for MLP input
        sequence = X[i - seq_len : i].flatten()
        Xs.append(sequence)
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_neural_network():
    """
    Train a neural network for stock price direction prediction
    """
    print("Loading and preparing data...")

    # Load data
    data = pd.read_csv("stock_data.csv")

    # Check data
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Target distribution: {data['Target'].value_counts()}")

    # Prepare features
    features = ["Open", "High", "Low", "Close", "Volume"]
    X = data[features].values
    y = data["Target"].values

    # Add technical indicators
    print("Adding technical indicators...")

    # Simple Moving Averages
    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()

    # Price ratios
    data["High_Low_Ratio"] = data["High"] / data["Low"]
    data["Close_Open_Ratio"] = data["Close"] / data["Open"]

    # Volatility (rolling standard deviation)
    data["Volatility"] = data["Close"].rolling(window=5).std()

    # Volume ratio
    data["Volume_MA"] = data["Volume"].rolling(window=5).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Updated features with technical indicators
    enhanced_features = features + [
        "SMA_5",
        "SMA_20",
        "High_Low_Ratio",
        "Close_Open_Ratio",
        "Volatility",
        "Volume_Ratio",
    ]

    X_enhanced = data[enhanced_features].values
    y_enhanced = data["Target"].values

    print(f"Enhanced data shape: {X_enhanced.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enhanced)

    # Create time series sequences (smaller window for MLP)
    seq_len = 5  # Smaller sequence length
    print(f"Creating sequences with window size {seq_len}...")

    X_seq, y_seq = create_sequences(X_scaled, y_enhanced, seq_len)
    print(f"Sequence data shape: X={X_seq.shape}, y={y_seq.shape}")

    # Split data chronologically (important for time series)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Testing target distribution: {np.bincount(y_test)}")

    # Train multiple models and compare
    models = {
        "MLP_Small": MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "MLP_Medium": MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "MLP_Large": MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate="adaptive",
        ),
    }

    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"{name} - Training Accuracy: {train_acc:.4f}")
        print(f"{name} - Testing Accuracy: {test_acc:.4f}")

        # Store results
        results[name] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_pred": test_pred,
        }

        # Track best model
        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name

    # Detailed evaluation of best model
    print(f"\n=== Best Model: {best_name} ===")
    print(f"Best Test Accuracy: {best_score:.4f}")

    best_pred = results[best_name]["test_pred"]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, best_pred))

    # Feature importance (if available)
    try:
        # For neural networks, we can't get feature importance directly
        # But we can analyze prediction confidence
        pred_proba = best_model.predict_proba(X_test)
        confidence = np.max(pred_proba, axis=1)
        print(
            f"\nPrediction Confidence - Mean: {confidence.mean():.4f}, Std: {confidence.std():.4f}"
        )
    except:
        pass

    # Save the best model and scaler
    try:
        joblib.dump(best_model, "best_neural_model.pkl")
        joblib.dump(scaler, "neural_scaler.pkl")

        # Save model parameters for future use
        model_info = {
            "model_name": best_name,
            "test_accuracy": best_score,
            "features": enhanced_features,
            "seq_len": seq_len,
            "scaler_type": "StandardScaler",
        }
        joblib.dump(model_info, "model_info.pkl")

        print(f"\nModel saved successfully!")
        print(f"Files saved: best_neural_model.pkl, neural_scaler.pkl, model_info.pkl")

    except Exception as e:
        print(f"Error saving model: {e}")

    return best_model, scaler, results


def make_prediction(
    model_path="best_neural_model.pkl",
    scaler_path="neural_scaler.pkl",
    info_path="model_info.pkl",
    data_path="stock_data.csv",
):
    """
    Load model and make predictions on recent data
    """
    try:
        # Load model components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        model_info = joblib.load(info_path)

        print(f"Loaded model: {model_info['model_name']}")
        print(f"Model accuracy: {model_info['test_accuracy']:.4f}")

        # Load and prepare recent data
        data = pd.read_csv(data_path)

        # Add technical indicators (same as training)
        data["SMA_5"] = data["Close"].rolling(window=5).mean()
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["High_Low_Ratio"] = data["High"] / data["Low"]
        data["Close_Open_Ratio"] = data["Close"] / data["Open"]
        data["Volatility"] = data["Close"].rolling(window=5).std()
        data["Volume_MA"] = data["Volume"].rolling(window=5).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]

        data.dropna(inplace=True)

        # Get recent data for prediction
        features = model_info["features"]
        seq_len = model_info["seq_len"]

        X_recent = data[features].values[-seq_len:]
        X_scaled = scaler.transform(X_recent)
        X_pred = X_scaled.flatten().reshape(1, -1)

        # Make prediction
        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0]

        print(f"\nPrediction for next trading day:")
        print(f"Direction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"Confidence: {max(probability):.4f}")
        print(f"Probabilities - DOWN: {probability[0]:.4f}, UP: {probability[1]:.4f}")

        return prediction, probability

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None


if __name__ == "__main__":
    print("=== Neural Network Stock Prediction ===")
    print("Training alternative neural network model...\n")

    # Train model
    try:
        model, scaler, results = train_neural_network()
        print("\n" + "=" * 50)
        print("Training completed successfully!")

        # Test prediction
        print("\nTesting prediction on recent data...")
        prediction, probability = make_prediction()

    except Exception as e:
        print(f"Error in training: {e}")
        import traceback

        traceback.print_exc()
