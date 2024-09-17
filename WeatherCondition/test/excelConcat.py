import pandas as pd
import os
import random
from datetime import datetime
from datetime import timedelta


def get_randomNum():
    return random.randint(22,35)

def generate_dates(n):
    dates = []
    for _ in range(n):
        date = (datetime.now() - timedelta(days=_ + 1)).strftime('%Y-%m-%d')
        dates.append(date)
    return dates

excel_folder = "data/excel/"
excel_path = []
for file in os.listdir(excel_folder):
    file_path = os.path.join(excel_folder, file)
    excel_path.append(file_path)

# 创建一个空的DataFrame，包含您想要的列名
df = pd.DataFrame()
# 读取所有Excel文件并将它们添加到一个列表中
dataframes = []
for path in excel_path:
    df1 = pd.read_excel(path, header=None)
    # 删除缺少一个数据的行
    df1 = df1.dropna(how='all', axis=0)
    df1 = df1.dropna(how='all', axis=1)
    # 重置索引
    df1 = df1.reset_index(drop=True)
    print(df1.columns)
    
    # df1.drop(labels=["D"], axis=1)
    dataframes.append(df1)
# 使用concat方法合并DataFrame列表
combined_df = pd.concat(dataframes)
combined_df.columns = ["timestamp", "AQI", "Range", "level", "avg_PM2.5", "avg_co2", "avg_formaldehyde", "avg_humidity", "o3", "NO3", "NO2"]
combined_df = combined_df.drop(["level","o3", "NO3", "NO2", "AQI", "Range", "timestamp"], axis=1)
combined_df["avg_co2"] = combined_df["avg_co2"].mul(6)
combined_df.insert(0, "id", [i for i in range(len(combined_df))])
combined_df.insert(4, "avg_temperature", [get_randomNum() for num in range(len(combined_df))])
combined_df.insert(1, "timestamp", [generate_dates(len(combined_df))[-(i+1)] for i in range(len(combined_df))])
# 保存合并后的DataFrame为CSV文件
combined_df.to_csv("data/CSV/merge.csv", index=False)
