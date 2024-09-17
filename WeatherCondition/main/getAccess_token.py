
import requests
import json

def main():
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=UkOt0GxNmbLdqPFflKgH3EZj&client_secret=C6mkurVJRFVNHxSbjDMfe2yVGRdWgTMD"
    
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    with open("data/response_json.txt", "w") as f:
        json_data = json.dumps(response.text)
        f.write(json_data)
    

if __name__ == '__main__':
    main()