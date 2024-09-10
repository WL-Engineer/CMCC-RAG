"""  
@author: VL
@time: 2024/9/3 10:59  
@file: streamlit_app.py  
@project: RAGandLangChain  
@description: 构建应用程序
"""
import os
import sys
import streamlit as st
import toml
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from ollamatest.zhipuai_enbedding import ZhipuAIEmbeddings
from ollamatest.zhipuai_llm import ZhipuAILLM

sys.path.append("ollamatest")

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

config = toml.load('..\\config.toml')
zhipuai_api_key = config['database']["ZHIPUAI_API_KEY"]

def generate_response(input_text, zhipuai_api_key):
    llm = ZhipuAILLM(model=MODEL, temperature=TEMPERATURE, api_key=zhipuai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output


# st.title('🦜🔗 张大宝大模型应用开发展示')
# zhipuai_api_key = st.sidebar.text_input('ZhipuAI API Key', type='password')
# ### 此处需要增加相关判断条件，及返回结果的处理过程
# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     # if not zhipuai_api_key.startswith('93-'):
#     #     st.warning('Please enter your zhipuAI API key!', icon='⚠')
#     # if submitted and zhipuai_api_key.startswith('93-'):
#     #     generate_response(text)
#     if submitted:
#         generate_response(text)

def get_vectordb():
    embedding = ZhipuAIEmbeddings()
    persist_directory = 'vector_chroma'
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
#
# cc = get_qa_chain("西瓜书","931f8779ed73460fa8852cb179ffcdbf.EkMJK7s7nXzaE6M6")
# print(cc)

def main():
    st.title('🦜🔗 科技创新制度应用开发')
    zhipu_api_key = st.sidebar.text_input('ZhipuAI API Key', type='password')
    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    if 'messages' not in st.session_state:
        st.session_state.messsages = []

    messages = st.container(height=200)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messsages.append({"role": "user", "text": prompt})
        if selected_method == "None":
            answer = "001" + generate_response(prompt, zhipu_api_key)
        elif selected_method == "qa_chain":
            answer = "002" + get_qa_chain(prompt, zhipu_api_key)
        elif selected_method == "chat_qa_chain":
            answer = "003" + get_chat_qa_chain(prompt, zhipu_api_key)

        # # 调用 respond函数来获取回答
        # answer = generate_response(prompt, zhipuai_api_key)
        if answer is not None:
            # 将LLM的回答添加到历史对话中
            st.session_state.messsages.append({"role": "assistant", "text": answer})
        for message in st.session_state.messsages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])


if __name__ == "__main__":
    main()
