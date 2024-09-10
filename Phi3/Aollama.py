import requests
import json

url = 'http://127.0.0.1:11434/api/generate'
# data = {
#     "model": "phi3:latest",
#     "messages": [
#         {"role": "user",
#          "content": "你是谁"},
#         {"role": "user",
#          "content": "你从哪里来"},
#         {"role": "user",
#          "content": "你能干什么"}
#     ]
# }
data = {
    # "model": "qwen2:0.5b",
    "model": "phi3:latest",
    "prompt": "如何在win通过命令行停止ollama正在运行的模型",
#    "format": "json"
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

    print(response.text)
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
