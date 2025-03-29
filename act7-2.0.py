import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from neuralprophet import NeuralProphet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
import yfinance as yf
import torch
from neuralprophet.configure import ConfigSeasonality

torch.serialization.add_safe_globals({'neuralprophet.configure.ConfigSeasonality': ConfigSeasonality})

# ======== Funciones de carga ========
@st.cache_data
def load_symbols():
    df = pd.read_csv("datasets/symbols_valid_meta.csv")
    return df[['Symbol', 'Security Name']].dropna()

@st.cache_data
def load_yfinance_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        raise ValueError("No se pudieron obtener datos desde Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'Close'.")
    return df.dropna(subset=['Close'])

# ======== Indicadores Técnicos ========
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger(data, window=20):
    sma = calculate_sma(data, window)
    std = data['Close'].rolling(window=window).std()
    upper = sma + 2*std
    lower = sma - 2*std
    return upper, lower

# ======== Gráfico ========
def plot_chart(df, chart_type, indicators):
    fig = go.Figure()
    df = df.dropna(subset=['Close'])

    if chart_type == 'line':
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    elif chart_type == 'area':
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], fill='tozeroy', mode='lines', name='Close'))
    elif chart_type == 'bar':
        fig.add_trace(go.Bar(x=df['Date'], y=df['Close'], name='Close'))
    elif chart_type == 'candlestick' and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Candlestick'))

    if 'SMA' in indicators:
        sma = calculate_sma(df, 20).dropna()
        fig.add_trace(go.Scatter(x=df['Date'].iloc[-len(sma):], y=sma, name='SMA'))
    if 'EMA' in indicators:
        ema = calculate_ema(df, 20).dropna()
        fig.add_trace(go.Scatter(x=df['Date'].iloc[-len(ema):], y=ema, name='EMA'))
    if 'MACD' in indicators:
        macd, signal = calculate_macd(df)
        fig.add_trace(go.Scatter(x=df['Date'], y=macd, name='MACD'))
        fig.add_trace(go.Scatter(x=df['Date'], y=signal, name='Signal'))
    if 'RSI' in indicators:
        rsi = calculate_rsi(df).dropna()
        fig.add_trace(go.Scatter(x=df['Date'].iloc[-len(rsi):], y=rsi, name='RSI'))
    if 'Bollinger Bands' in indicators:
        upper, lower = calculate_bollinger(df)
        upper, lower = upper.dropna(), lower.dropna()
        fig.add_trace(go.Scatter(x=df['Date'].iloc[-len(upper):], y=upper, name='Upper Band'))
        fig.add_trace(go.Scatter(x=df['Date'].iloc[-len(lower):], y=lower, name='Lower Band'))

    fig.update_layout(title='Gráfico con Indicadores', xaxis_title='Fecha', yaxis_title='Precio')
    return fig

# ======== Gráfico con Subgráficos y Comparación ========
def plot_comparison_chart(df1, df2, symbol1, symbol2):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"Precio de {symbol1}", f"Precio de {symbol2}"))

    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Close'], name=symbol1), row=1, col=1)
    fig.add_trace(go.Scatter(x=df2['Date'], y=df2['Close'], name=symbol2), row=2, col=1)

    fig.update_layout(height=600, title_text="Comparación de Precios de Acciones",
                      xaxis_title="", yaxis_title="Precio")
    return fig

# ======== Visualización de Predicciones ========
def display_forecast_chart(predictions, model_name):
    df = pd.DataFrame({
        "Día": list(range(1, len(predictions)+1)),
        "Predicción": predictions.flatten() if predictions.ndim > 1 else predictions
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Día'], y=df['Predicción'], mode='lines+markers', name='Predicción'))
    fig.update_layout(title=f"Predicción de Precios - {model_name}", xaxis_title="Día", yaxis_title="Precio")
    st.plotly_chart(fig)

# ======== Modelos de Predicción ========
def arima_model(df):
    model = ARIMA(df['Close'], order=(5, 1, 0)).fit()
    return model.forecast(30)

def neural_prophet_model(df):
    from neuralprophet.configure import ConfigSeasonality
    import torch
    torch.serialization.add_safe_globals({'neuralprophet.configure.ConfigSeasonality': ConfigSeasonality})
    
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = NeuralProphet()
    model.fit(df_prophet, freq='D')
    future = model.make_future_dataframe(df_prophet, periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat1']]


def lstm_model(df):
    if len(df) < 60:
        raise ValueError("Se requieren al menos 60 registros para aplicar el modelo LSTM.")
    df = df[['Close']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    pred = model.predict(X[-30:])
    return scaler.inverse_transform(pred)

# ======== Descripción de Modelos ========
def model_description(model_choice):
    if model_choice == 'ARIMA':
        return """
        **ARIMA (Autoregressive Integrated Moving Average)** es un modelo estadístico útil para pronosticar series temporales que muestran tendencia o estacionalidad. Captura patrones de autocorrelación y variaciones temporales.
        """
    elif model_choice == 'NeuralProphet':
        return """
        **NeuralProphet** es una extensión de Facebook Prophet que usa redes neuronales para capturar patrones complejos, estacionales y no lineales en series temporales.
        """
    elif model_choice == 'LSTM':
        return """
        **LSTM (Long Short-Term Memory)** es una red neuronal recurrente especializada en aprender dependencias a largo plazo, ideal para series temporales con comportamientos no lineales.
        """
    return ""

# ======== Interfaz Streamlit Principal ========
st.title("\U0001F4CA Dashboard de Trading con Indicadores y Modelos")
symbols_df = load_symbols()
symbol = st.selectbox("Selecciona un símbolo:", symbols_df['Symbol'])
symbol_name = symbols_df[symbols_df['Symbol'] == symbol]['Security Name'].values[0]
st.markdown(f"**Símbolo seleccionado:** {symbol} — *{symbol_name}*")

chart_type = st.selectbox("Tipo de gráfico:", ['line', 'area', 'bar', 'candlestick'])
indicators = st.multiselect("Indicadores técnicos:", ['SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands'])

start_date, end_date = st.date_input("Selecciona el rango de fechas:", value=(datetime(2015, 1, 1), datetime.today()))

try:
    df_data = load_yfinance_data(symbol, period="max")
    df_data = df_data[(df_data['Date'] >= pd.to_datetime(start_date)) & (df_data['Date'] <= pd.to_datetime(end_date))]

    if len(df_data) >= 2:
        latest = df_data['Close'].iloc[-1]
        prev = df_data['Close'].iloc[-2]
        change = latest - prev
        change_text = f"Variación más reciente: {change:+.2f}"
        change_color = 'green' if change > 0 else 'red' if change < 0 else 'black'
        st.markdown(f"<span style='color:{change_color}'>{change_text}</span>", unsafe_allow_html=True)

    # Gráfico principal
    st.plotly_chart(plot_chart(df_data, chart_type, indicators))

    model_choice = st.selectbox("Selecciona un modelo:", ['ARIMA', 'NeuralProphet', 'LSTM'])
    st.markdown(model_description(model_choice))

    if model_choice == 'ARIMA':
        forecast = arima_model(df_data)
        display_forecast_chart(forecast, "ARIMA")
    elif model_choice == 'NeuralProphet':
        forecast_df = neural_prophet_model(df_data)
        forecast = forecast_df['yhat1'].values
        display_forecast_chart(forecast, "NeuralProphet")
    elif model_choice == 'LSTM':
        forecast = lstm_model(df_data)
        display_forecast_chart(forecast, "LSTM")

except Exception as e:
    st.error(f"Error al cargar datos: {e}")

# ======== Comparador de símbolos ========
st.subheader("\U0001F4C8 Comparación de dos símbolos")
col1, col2 = st.columns(2)
with col1:
    symbol1 = st.selectbox("Primer símbolo:", symbols_df['Symbol'], key="s1")
with col2:
    symbol2 = st.selectbox("Segundo símbolo:", symbols_df['Symbol'], key="s2")

try:
    df1 = load_yfinance_data(symbol1, period="max")
    df2 = load_yfinance_data(symbol2, period="max")

    df1 = df1[(df1['Date'] >= pd.to_datetime(start_date)) & (df1['Date'] <= pd.to_datetime(end_date))]
    df2 = df2[(df2['Date'] >= pd.to_datetime(start_date)) & (df2['Date'] <= pd.to_datetime(end_date))]

    st.plotly_chart(plot_comparison_chart(df1, df2, symbol1, symbol2))

except Exception as e:
    st.error(f"Error al cargar o comparar datos: {e}")
