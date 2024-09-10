"""
@author: VL
@time: 2024/9/3 9:24
@file: datawrite.py
@project: RAGandLangChain
@description: 测试OPENAI的CHATGPT   API，目前未通过()
"""
import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

MODEL = "gpt-3.5-turbo-1106"

_ = load_dotenv(find_dotenv())
# print(os.environ["OPENAI_API_KEY"])

# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are a helpful asssistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(completion.choices[0].message.content)
