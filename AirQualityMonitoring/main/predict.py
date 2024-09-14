import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

# 创建数据集
def create_dataset(dataset, look_back=1): 
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])  
    return np.array(X), np.array(Y)

def predict_and_save(model, scaler, airQualityDF, n_steps_ahead=10):
    # 选择特征列
    feature_columns = ['avg_co2', 'avg_pm25', 'avg_formaldehyde', 'avg_temperature', 'avg_humidity']
    airQualityDF_features = airQualityDF[feature_columns]
    
    # 对最后一个样本进行归一化
    last_sample_normalized = scaler.transform(airQualityDF_features.iloc[-1:].values.reshape(1, -1))
    last_sample = last_sample_normalized.reshape(1, 1, len(feature_columns))
    
    # 预测未来的数据点
    predicted_values_normalized = np.zeros((n_steps_ahead, len(feature_columns)))
    for i in range(n_steps_ahead):
        predicted_value_normalized = model.predict(last_sample)
        predicted_values_normalized[i, :] = predicted_value_normalized
        # 更新最后一个样本以进行下一次预测
        last_sample = np.concatenate((last_sample[:, 1:, :], predicted_value_normalized.reshape(1, 1, len(feature_columns))), axis=1)
    
    # 逆标准化预测结果
    predicted_values = scaler.inverse_transform(predicted_values_normalized)
    
    # 生成新的时间戳
    last_timestamp = datetime.strptime(airQualityDF['timestamp'].iloc[-1], "%Y/%m/%d %H:%M")
    new_timestamps = [last_timestamp + timedelta(minutes=i + 1) for i in range(n_steps_ahead)]
    
    # 插入预测行
    for i, prediction in enumerate(predicted_values):
        formatted_time = new_timestamps[i].strftime("%Y/%m/%d %H:%M")
        new_row = {
            'id': airQualityDF['id'].iloc[-1] + i + 1,
            'timestamp': formatted_time,
            'avg_co2': prediction[0],
            'avg_pm25': prediction[1],
            'avg_formaldehyde': prediction[2],
            'avg_temperature': prediction[3],
            'avg_humidity': prediction[4]
        }
        airQualityDF = pd.concat([airQualityDF, pd.DataFrame([new_row])], ignore_index=True)
    
    # 将更新后的 DataFrame 保存为 JSON 文件
    airQualityDF.to_json("data/json/airQualityData.json", orient='records', lines=True)

def main():
    airQualityDF = pd.read_csv("data/CSV/AirQualityData.csv")
    model = load_model("model/air_quality_model.h5")
    with open("model/scaler.save", "rb") as file:
        scaler = pickle.load(file)
    predict_and_save(model, scaler, airQualityDF)

if __name__ == '__main__':
    main()