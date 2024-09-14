import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from datetime import datetime


def is_fill(time):
    fillnaCount = time.isnull().sum()
    if fillnaCount != 0:
        print(f"时间戳存在缺失值数量：{fillnaCount}")
        time.fillna(method='ffill', inplace=True)  

def modelTrainAndPrediction(airQualityDF, feature_columns):
# data_normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(airQualityDF[feature_columns])

    scaled_features_df = pd.DataFrame(scaled_features, index=airQualityDF.index, columns=feature_columns)
#splitTrainset
    n_steps = 2
    n_features = scaled_features_df.shape[1]

    # 将数据集划分为训练集和测试集
    train_size = int(len(scaled_features_df) * 0.8)
    train, test = scaled_features_df[0:train_size].values, scaled_features_df[train_size:len(scaled_features_df)].values
  # 转换为NumPy数组
    look_back = n_steps
    trainX, trainY = create_dataset(train, look_back)

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, n_features)))
    model.add(Dense(n_features))  # 修改输出层以匹配特征列的数量
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(trainX, trainY, epochs=1000, verbose=2)
    # 假设我们要预测未来10个时间步长的 'avg_co2' 值
    n_steps_ahead = 10
    predicted_values_normalized = np.zeros((n_steps_ahead, n_features))

    # 使用测试集的最后一个样本作为起始点
    last_sample = test[-2:, :]  # 获取测试集的最后一个样本

    last_sample = last_sample.reshape(1, 2, n_features)
    # 更新last_sample，为下一次预测做准备
    for i in range(n_steps_ahead):
        predicted_value_normalized = model.predict(last_sample)
        predicted_values_normalized[i, :] = predicted_value_normalized
        last_sample = np.concatenate((last_sample[:, 1:, :], predicted_value_normalized.reshape(1, 1, n_features)), axis=1)

    # 逆标准化预测结果
    predicted_values = scaler.inverse_transform(predicted_values_normalized)

    # 打印预测结果
    for i, val in enumerate(predicted_values):
        print(f'Predicted values for step {i+1}: {val}')

    # 插入预测行
    last_id = airQualityDF['id'].iloc[-1]
    now = datetime.now()
    formatted_time = now.strftime("%Y/%m/%d %H:%M")  # 时间戳

    for i, prediction in enumerate(predicted_values):
        new_row = {
            'id': last_id + i + 1,
            'timestamp': formatted_time,
            'avg_co2': prediction[0],  
            'avg_pm25': prediction[1],  
            'avg_formaldehyde': prediction[2],  
            'avg_temperature': prediction[3],  
            'avg_humidity': prediction[4],  
        }
        airQualityDF = pd.concat([airQualityDF, pd.DataFrame([new_row])], ignore_index=True)

    # 将新的预测数据添加到原始 DataFrame
    

    # 将更新后的 DataFrame 保存为 JSON 文件
    airQualityDF.to_json("data/json/airQualityData.json", orient='records', lines=True)

# 创建数据集
def create_dataset(dataset, look_back=1): 
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])  
    return np.array(X), np.array(Y)


def main():
    airQualityDF = pd.read_csv("data/CSV/AirQualityData.csv")
    time = airQualityDF["timestamp"]
    feature_columns = ['avg_co2', 'avg_pm25', 'avg_formaldehyde', 'avg_temperature', 'avg_humidity']
    is_fill(time=time)
    modelTrainAndPrediction(feature_columns= feature_columns, airQualityDF=airQualityDF)
    

if __name__ == '__main__':
    main()