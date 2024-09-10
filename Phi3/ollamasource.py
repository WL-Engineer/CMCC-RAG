import ollama

host = "127.0.0.1"
port = "11434"
client = ollama.Client(host=f"http://{host}:{port}")
res = client.chat(
    model="phi3:latest",
    messages=[{
        "role": "user",
        "content": "你是谁"
    },{
        "role": "user",
        "content": "有什么工作是你能做的"
    }],
    stream= True,
)
print(res['message']['content'])