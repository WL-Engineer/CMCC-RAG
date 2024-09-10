import os
from dotenv import load_dotenv, find_dotenv
from zhipuai_llm import ZhipuAILLM
"""  
@author: VL
@time: 2024/9/3 9:24  
@file: testchatglm.py  
@project: RAGandLangChain  
@description: 测试通过longchain调用智谱大模型
"""
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
TEMPERATURE = 0.1

_ = load_dotenv(find_dotenv())
api_key = os.environ["ZHIPUAI_API_KEY"]

zhipuai_model = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key=api_key)
response = zhipuai_model.invoke(input="你好，请你自我介绍一下你的具体模型和如何使用langchain调用你")
print(response)
