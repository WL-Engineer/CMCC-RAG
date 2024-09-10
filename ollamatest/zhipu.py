"""
@time: 2024/9/3 9:24
@file: zhipu.py
@project: RAGandLangChain
@description: []使用智谱AI提供的类，并测试智谱API通过
"""
import os

import toml
from zhipuai import ZhipuAI

from ollamatest.datawrite import insertdata

config = toml.load('..\\config.toml')
client = ZhipuAI(
    api_key=config['database']["ZHIPUAI_API_KEY"]
)

# print(os.environ["ZHIPUAI_API_KEY"])

#### 语言模型收费标准
# # flash 免费
# MODEL = "glm-4-flash"
# # plus 0.05/千token   高性能旗舰
MODEL = "glm-4-plus"
# # Air  0.001/千token  高性价比
# MODEL = "glm-4-air"
# # long 0.001/千token  超长输入
# MODEL = "glm-4-long"


# 采样温度，控制输出的随机性，必须为正数取值范围是：(0.0, 1.0)，不能等于 0，默认值为 0.95。值越大，会使输出更随机，更具创造性；值越小，输出会更加稳定或确定
TEMPERATURE = 0.7


def gen_glm_params(prompt):
    message = [{
        "role": "user",
        "content": prompt
    }]
    return message


def get_completion(prompt):
    message = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        temperature=TEMPERATURE
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"


# 使用分隔符(指令内容，使用 ``` 来分隔指令和待总结的内容)

prompt = f"你好，请介绍一下你自己"
data = get_completion(prompt)



print(data)
insertdata(MODEL, prompt, data)
print("写入成功")
