import streamlit as st

stock_ticker = ""
start_date = ""
end_date = ""

def intro():
    import streamlit as st

    st.write("Welcome To Our Basic Stock Predictor 👋")
    st.sidebar.success("Select An Option Above.")

    st.markdown(
        """
        Currently, this is still under production, and can only display basic visualization graphs, and a simple
        LSTM Model of the stock, however in the future, this can be further expanded depending on how this project
        wants to be further explored.
    """
    )




def visualizing_stock():
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    Ticker_Symbol = st.text_input("Enter value for Stock Ticker", value = "googl", help = "googl")
    Start_Date = st.text_input("Enter value for Starting Date", help = "1999-01-01", value = "1999-01-01")
    End_Date = st.text_input("Enter value for Ending Date", help = "2024-01-01", value = "2024-01-01")
    try:
        Data = yf.download(Ticker_Symbol, Start_Date, End_Date)
    except:
        st.warning("HO")
    
    #Display DataFrame
    Data = Data.reset_index()
    Data["Date"] = Data["Date"].dt.date
    st.header("Displaying The DataFrame of your Choice: ")
    st.dataframe(Data)

    #Closing Price
    st.header("Closing Price per Day: ")
    st.line_chart(Data, x = "Date", y = "Close")

    #Moving Average
    Data["MA60"] = Data.Close.rolling(60).mean()
    Data["MA250"] = Data.Close.rolling(250).mean()

    st.header("Moving Average: ")
    st.line_chart(Data, x = "Date", y = ["Close","MA60", "MA250"])

    #Daily Volume Chart
    st.header("Daily Volume Chart: ")
    st.line_chart(Data, x = "Date", y = "Volume")

    #Daily Return Chart
    Data["Daily_Return"] = Data["Close"].pct_change()
    st.header("Daily Return Chart: ")
    st.line_chart(Data, x = "Date", y = "Daily_Return")

    #Cummulative Return Chart
    Data["Cumulative_Return"] = (1 + Data["Daily_Return"]).cumprod() - 1
    st.header("Cumulative Return Chart: ")
    st.line_chart(Data, x = "Date", y = "Cumulative_Return")   


def lstm_stock():
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout

    Ticker_Symbol = st.text_input("Enter value for Stock Ticker", value = "googl", help = "googl")
    Start_Date = st.text_input("Enter value for Starting Date", help = "1999-01-01", value = "1999-01-01")
    End_Date = st.text_input("Enter value for Ending Date", help = "2024-01-01", value = "2024-01-01")
    try:
        Data = yf.download(Ticker_Symbol, Start_Date, End_Date)
    except:
        st.warning("HO")
    
    #Display DataFrame
    Data = Data.reset_index()
    Data["Date"] = Data["Date"].dt.date
    st.header("Displaying The DataFrame of your Choice: ")
    st.dataframe(Data)

    Close = Data["Close"]
    Close_Value = Close.values
    Close_Value = Close_Value.reshape(-1,1)
    Training_Data_Leng = math.ceil(len(Close_Value) * 0.7)
    scaler = MinMaxScaler(feature_range=(0,1))
    PriceData = scaler.fit_transform(Close_Value)
    X_train, Y_train = [], []
    Backcandles = 60
    TrainData = PriceData[:Training_Data_Leng]
    for i in range(Backcandles, len(TrainData)):
        X_train.append(TrainData[i - Backcandles : i, 0])
        Y_train.append(TrainData[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    Model = Sequential([
    LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1)),
    (Dropout(0.2)),
    LSTM((50)),
    (Dropout(0.2)),
    (Dense(32)),
    (Dense(1))
    ])
    Model.compile(optimizer = "adam", loss = "mean_squared_error")
    Model.fit(X_train,Y_train, batch_size = 32, epochs = 10)

    Test_Data = PriceData[Training_Data_Leng - Backcandles:, :]
    x_test, y_test = [], Close_Value[Training_Data_Leng:, :]
    for i in range(Backcandles, len(Test_Data)):
        x_test.append(Test_Data[i-Backcandles:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    Pred = Model.predict(x_test)
    Pred = scaler.inverse_transform(Pred)
    # RMSE = np.sqrt(np.mean(Pred- y_test) ** 2)
    TrainSet, ValidSet = Close[:Training_Data_Leng], Close[Training_Data_Leng:]
    ValidSet = pd.DataFrame(ValidSet)
    ValidSet["Prediction"] = Pred
    ValidSet.reset_index()


    st.header("Visualize Predicted Results ")
    st.line_chart(ValidSet,  y = ["Close", "Prediction"])

page_names_to_funcs = {
    "—": intro,
    "Visualize": visualizing_stock,
    "Predicting Stock": lstm_stock,
    # "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()