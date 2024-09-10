"""
@time: 2024/9/3 9:24
@file: testZhuEmbedding.py
@project: RAGandLangChain
@description: 单个词语生成智谱词向量函数及其测试
"""
import os
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI

from ollamatest.datawrite import vectorWrite

MODEL = "embedding-2"

_ = load_dotenv(find_dotenv())


def zhipu_embedding(text: str):
    api_key = os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return response


text = '要生成embedding的输入文本，字符'
response = zhipu_embedding(text=text)
print(f'response类型为：{type(response)}')
print(f'embedding类型为：{response.object}')
print(f'生成embedding的model为：{response.model}')
print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:1024]}')

# vectorWrite(MODEL, text, response.data[0].embedding[:1024])
