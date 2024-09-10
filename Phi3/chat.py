import requests
import json

url = 'http://127.0.0.1:11434/api/chat'
data = {
    "model": "phi3:latest",
    "messages": [
        {
            "role": "system",
            "content": "Your are a python developer."
        },
        {
            "role": "user",
            "content": "Help me generate a bubble algorithm"
        }
    ],
    "stream": False

}

headers = {
    'Content-Type': 'application'
}
response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    print("获取内容成功，请输出")
    # print(response.text)
    # print("--------------------------------------")
    # print(response.content)
    print("--------------------------------------")
    # 解析并打印响应内容
    response_str = response.content.decode('utf-8')
    json_strings = response_str.split('\n')
    json_strings = [s for s in json_strings if s]
    json_objects = [json.loads(s) for s in json_strings]
    for obj in json_objects:
        print(obj['response'], end='')
    print()
    print("--------------------------------------")
else:
    print("Failed to generate text:", response.status_code, response.text)
