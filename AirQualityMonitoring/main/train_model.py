import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle  # 导入pickle模块
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

def create_dataset(dataset, look_back=1): 
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])  
    return np.array(X), np.array(Y)

def train_model(airQualityDF, feature_columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(airQualityDF[feature_columns])

    scaled_features_df = pd.DataFrame(scaled_features, index=airQualityDF.index, columns=feature_columns)

    n_steps = 20
    n_features = scaled_features_df.shape[1]

    train_size = int(len(scaled_features_df) * 0.8)
    train, test = scaled_features_df[0:train_size].values, scaled_features_df[train_size:len(scaled_features_df)].values

    look_back = n_steps
    trainX, trainY = create_dataset(train, look_back)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')

    model.fit(trainX, trainY, epochs=1000, verbose=2)

    # 使用pickle保存scaler
    with open("model/scaler.save", "wb") as file:
        pickle.dump(scaler, file)

    return model

def main():
    airQualityDF = pd.read_csv("data/CSV/AirQualityData.csv")
    feature_columns = ['avg_co2', 'avg_pm25', 'avg_formaldehyde', 'avg_temperature', 'avg_humidity']
    model = train_model(airQualityDF, feature_columns)
    model.save("model/air_quality_model.h5")

if __name__ == '__main__':
    main()