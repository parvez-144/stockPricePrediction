import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch Data
def fetch_stock_data(ticker, period="5y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data found for ticker symbol. Please check the symbol and try again.")
        data['Average'] = (data['Open'] + data['Close']) / 2
        return data['Average'].dropna()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Train ARIMA Model on Trend
def train_arima_model(data, order=(5,1,0)):
    arima_model = ARIMA(data, order=order)
    arima_fit = arima_model.fit()
    trend_prediction = arima_fit.predict(start=0, end=len(data)-1)
    residual = data - trend_prediction  # Calculate residual for LSTM training
    return trend_prediction, residual, arima_fit

# Preprocess Data for LSTM (using residual component)
def preprocess_data_for_lstm(residual, sequence_length=30):
    scaler = MinMaxScaler()
    scaled_residual = scaler.fit_transform(residual.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_residual)):
        X.append(scaled_residual[i-sequence_length:i])
        y.append(scaled_residual[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Train LSTM Model on Residuals
def train_lstm_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model

# Hybrid Prediction Function
def hybrid_predict(ticker, arima_order=(5,1,0), lstm_sequence_length=30):
    # Fetch and preprocess data
    data = fetch_stock_data(ticker)
    if data is None:
        print("Hybrid prediction aborted due to data fetching error.")
        return None
    
    # ARIMA for trend
    trend_prediction, residual, arima_fit = train_arima_model(data, order=arima_order)
    
    # LSTM for residual
    X, y, scaler = preprocess_data_for_lstm(residual, sequence_length=lstm_sequence_length)
    lstm_model = train_lstm_model(X, y)
    
    # Forecast future trend with ARIMA
    future_trend = arima_fit.forecast(steps=30)  # 30-day forecast as an example
    
    # Forecast residuals with LSTM
    last_sequence = X[-1]  # Last available sequence for LSTM prediction
    future_residuals = []
    for _ in range(30):  # Predicting 30 days ahead
        lstm_pred = lstm_model.predict(last_sequence.reshape(1, lstm_sequence_length, 1))[0][0]
        future_residuals.append(lstm_pred)
        last_sequence = np.append(last_sequence[1:], [[lstm_pred]], axis=0)
    
    # Inverse scale LSTM predictions
    future_residuals = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1)).flatten()
    
    # Final hybrid prediction
    hybrid_prediction = future_trend + future_residuals
    
    # Visualize results
    visualize_results(data, trend_prediction, residual, future_trend, future_residuals, hybrid_prediction)
    
    return hybrid_prediction

# Visualization function for comparison
def visualize_results(data, trend_prediction, residual, future_trend, future_residuals, hybrid_prediction):
    plt.figure(figsize=(14, 12))
    
    # Historical Data
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Historical Stock Price', color='blue')
    plt.title('Historical Stock Price')
    plt.legend()
    
    # ARIMA Trend Prediction
    plt.subplot(4, 1, 2)
    plt.plot(data, label='Actual Price', color='blue', alpha=0.5)
    plt.plot(range(len(data), len(data) + 30), future_trend, label='ARIMA Trend Forecast', color='orange')
    plt.title('ARIMA Trend Prediction')
    plt.legend()
    
    # LSTM Residual Prediction
    plt.subplot(4, 1, 3)
    plt.plot(residual, label='Residuals from ARIMA', color='red', alpha=0.5)
    plt.plot(range(len(data), len(data) + 30), future_residuals, label='LSTM Residual Forecast', color='purple')
    plt.title('LSTM Residual Prediction')
    plt.legend()
    
    # Hybrid Prediction (ARIMA + LSTM)
    plt.subplot(4, 1, 4)
    plt.plot(data, label='Historical Price', color='blue', alpha=0.5)
    plt.plot(range(len(data), len(data) + 30), hybrid_prediction, label='Hybrid Forecast (ARIMA + LSTM)', color='green')
    plt.title('Hybrid Prediction (ARIMA + LSTM)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
ticker = 'ASHOKLEY.NS'  # Example company
hybrid_forecast = hybrid_predict(ticker)
if hybrid_forecast is not None:
    print("Hybrid Forecast:", hybrid_forecast)
else:
    print("No forecast generated due to data fetching issue.")