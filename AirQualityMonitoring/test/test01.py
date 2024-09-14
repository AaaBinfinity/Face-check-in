import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

airQualityDF = pd.read_csv("data/AirQualityData.csv")
time = airQualityDF["timestamp"]
fillnaCount = time.isnull().sum()
if fillnaCount != 0:
    print(f"时间戳存在缺失值数量：{fillnaCount}")
    time.fillna(method='ffill', inplace=True)  # 修改为正确的参数
feature_columns = ['avg_co2', 'avg_pm25', 'avg_formaldehyde', 'avg_temperature', 'avg_humidity']
target_column = 'avg_co2'

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(airQualityDF[feature_columns])

scaled_features_df = pd.DataFrame(scaled_features, index=airQualityDF.index, columns=feature_columns)

# 假设我们要预测 'avg_co2'，且我们使用60个时间步长
n_steps = 1
n_features = scaled_features_df.shape[1]

# 将数据集划分为训练集和测试集
train_size = int(len(scaled_features_df) * 0.8)
test_size = len(scaled_features_df) - train_size
train, test = scaled_features_df[0:train_size].values, scaled_features_df[train_size:len(scaled_features_df)].values  # 转换为NumPy数组

# 创建数据集
def create_dataset(dataset, look_back=1): 
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = n_steps
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(trainX, trainY, epochs=5, verbose=2)

# 假设我们要预测未来10个时间步长的 'avg_co2' 值
n_steps_ahead = 10
predicted_values_normalized = np.zeros((n_steps_ahead, n_features))

# 使用测试集的最后一个样本作为起始点
last_sample = test[-2:, :]  # 获取测试集的最后一个样本

last_sample = last_sample.reshape(1, 2, n_features)  # 重塑为(1,1,5)
print(last_sample)
# 更新last_sample，为下一次预测做准备
for i in range(n_steps_ahead):
    predicted_value_normalized = model.predict(last_sample)
    predicted_values_normalized[i, :] = predicted_value_normalized
    last_sample = np.concatenate((last_sample[:, 1:, :], predicted_values_normalized.reshape(1, n_steps_ahead, n_features)), axis=1)

# 逆标准化预测结果
predicted_values = scaler.inverse_transform(predicted_values_normalized)[:,0]

# 打印预测结果
for i, val in enumerate(predicted_values):
    print(f'Predicted avg_co2 value for step {i+1}: {val}')