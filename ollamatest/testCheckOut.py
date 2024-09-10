"""  
@author: VL
@time: 2024/9/3 9:24  
@file: testCheckOut.py
@project: RAGandLangChain  
@description: 从数据库中检索数据，构建LLM大模型 最终生成 检索问答链
"""
import os
import sys

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from ollamatest.zhipuai_enbedding import ZhipuAIEmbeddings
from ollamatest.zhipuai_llm import ZhipuAILLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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

sys.path.append("../")
_ = load_dotenv(find_dotenv())
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]
embedding = ZhipuAIEmbeddings()
persist_directory = '../vector_chroma'
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding
                  )
print(f"向量库中存储的数量:{vectordb._collection.count()}")
# question = "政策"
# docs = vectordb.similarity_search(question, k=3)
# print(f"检索到的内容数：{len(docs)}")
# for i, doc in enumerate(docs):
#     print(f"检索到的第{i}个内容：\n{doc.page_content}", end="\n-----------------------------------------------------\n")
# mmr_docs = vectordb.max_marginal_relevance_search(question, k=3)
# for i, sim_doc in enumerate(mmr_docs):
#     print(f"MMR检索到第{i}个内容：\n{sim_doc.page_content[:200]}",end="\n------------------------------------------------------------\n")
llm = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key=zhipuai_api_key)
# response = llm.invoke(input="你好，请你自我介绍一下你的具体模型")
# print(response)

# 创建模板
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["contest", "question"], template=template)

"""
创建基于模板的检索链
创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：
llm：指定使用的 LLM
指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
"""
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
# question_1 = "中国移动核心能力有哪些"
# question_2 = "移动内部创新分为几部分"
# result = qa_chain.invoke({"query": question_1})
# # result = qa_chain({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])
# propmt_template = f"请回答以下问题：{question_1}"
# print(llm.invoke(propmt_template))

#构建历史消息记录列表，实现历史回答
#关于更多的 Memory 的使用，包括保留指定对话轮数、保存指定 token 数量、保存历史对话的总结摘要等内容，请参考 langchain 的 Memory 部分的相关文档。
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与promopt 的输入变量保持一致
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)


qa = ConversationalRetrievalChain.from_llm(llm,
                                           retriever=vectordb.as_retriever(),
                                           memory=memory
                                           )
print("\n---------------------------------------------------------------------------------\n")
question = "网络包含哪几层？"
result = qa.invoke({"question": question})
print(result['answer'])
print("\n---------------------------------------------------------------------------------\n")
question = "传输层传输的信息结构是什么"
result = qa.invoke({"question": question})
print(result['answer'])
print("\n---------------------------------------------------------------------------------\n")
question = "网络安全都需要注意哪些方面？"
result = qa.invoke({"question": question})
print(result['answer'])
print("\n---------------------------------------------------------------------------------\n")
question = "网络工程师需要具备的技能有哪些"
result = qa.invoke({"question": question})
print(result['answer'])


