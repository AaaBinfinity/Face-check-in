# https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=UkOt0GxNmbLdqPFflKgH3EZj&client_secret=RPl4wyHRwTHLagPD3PLIFmuF4s3E9tDs&
# access_token = 24.5cb6933a20846d8461320e006b503a75.2592000.1724378574.282335-98146135
# encoding:utf-8

import requests
import base64
import os
from time import sleep
'''
表格文字识别
'''


request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/table"
# 二进制方式打开图片文件
folder_path = 'data/img'
for filename in os.listdir(folder_path):
    file = os.path.join(folder_path, filename)
    filename_without_extension = os.path.splitext(filename)[0]
    with open(file, 'rb') as f:
        img = base64.b64encode(f.read())
    params = {"image":img,"return_excel":"true"}
    access_token = '[24.3c0ea4138208fe27dd4fd5cb8140d796.2592000.1729063853.282335-98146135]'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        # with open('resources/excel/'+filename_without_extension+'.xlsx','wb') as f:
        #     f.write(base64.b64encode(response.content))
        json_content = response.json()
        excel_content_base64 = json_content["excel_file"]
        excel_content = base64.b64decode(excel_content_base64)
        with open(f'data/excel/{filename_without_extension}.xlsx', 'wb') as f:
            f.write(excel_content)
        
    sleep(2)