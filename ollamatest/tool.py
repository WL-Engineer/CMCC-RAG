"""  
@author: VL
@time: 2024/9/3 12:27  
@file: tool.py  
@project: RAGandLangChain  
@description: 检索问答链的工具类
"""
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from ollamatest.zhipuai_enbedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma

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
TEMPERATURE = 0.1
_ = load_dotenv(find_dotenv())
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]

def get_vectordb():
    embedding = ZhipuAIEmbeddings()
    persist_directory = '../vector_chroma'
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding
                      )
    return vectordb


# 带有历史记录的问答链
def get_chat_qa_chain(question: str, zhipuai_api_key: str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key=zhipuai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与promopt 的输入变量保持一致
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    qa = ConversationalRetrievalChain.from_llm(llm,
                                               retriever=vectordb.as_retriever(),
                                               memory=memory
                                               )
    result = qa.invoke({"question": question})
    return result['answer']
#
# print("\n---------------------------------------------------------------------------------\n")
# question = "我可以学习到关于低空经济的知识吗？"
# print(get_chat_qa_chain(question=question,zhipuai_api_key='931f8779ed73460fa8852cb179ffcdbf.EkMJK7s7nXzaE6M6'))
#
# print("\n---------------------------------------------------------------------------------\n")
# question = "请告诉我关于应用场景的相关知识"
# print(get_chat_qa_chain(question=question,zhipuai_api_key='931f8779ed73460fa8852cb179ffcdbf.EkMJK7s7nXzaE6M6'))





# 不带有历史记录的问答链
def get_qa_chain(question: str, zhipuai_api_key: str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key=zhipuai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["contest", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain.invoke({"query": question})
    return result["result"]
print(get_qa_chain("你好，你有检索到知识库吗", "931f8779ed73460fa8852cb179ffcdbf.EkMJK7s7nXzaE6M6"))