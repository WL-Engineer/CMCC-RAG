"""  
@author: VL
@time: 2024/9/3 11:30  
@file: tttt.py  
@project: RAGandLangChain  
@description: 这里是文件的描述信息  
"""
from ollamatest.zhipuai_llm import ZhipuAILLM

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
TEMPERATURE = 0.9

llm = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key="931f8779ed73460fa8852cb179ffcdbf.EkMJK7s7nXzaE6M6")
print(llm.invoke(input="你好"))