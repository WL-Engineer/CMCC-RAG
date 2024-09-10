"""
@author: VL
@time: 2024/9/3 9:24
@file: vector_library_construct.py
@project: RAGandLangChain
@description: 根据目录读取所有的页面，并采用智谱向量化API，将所有数据向量化，储存在本地文件夹中，
并根据两种不同的方式检测其匹配度
"""
import os
import re
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zhipuai_enbedding import ZhipuAIEmbeddings

from langchain_community.vectorstores import Chroma

# 知识库中单段文本长度
CHUNK_SIZE = 500
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 100

# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

file_paths = []
folder_path = '../fileSource'
# 遍历folder_path目录及其子目录将文件名称加入到列表中
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:10])
# 遍历文件路径并把实例化的loader存放在loaders里,文件格式仅为md或者pdf
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    # 下面这行可以不要，处理md文档
    else:
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 将所有的页面加入到text中
texts = []
for loader in loaders:
    texts.extend(loader.load())
print("一共有多少页"+str(len(texts)))
# 对文档进行处理，删除空格，调整句式等
# pdf_page = texts[15]
# pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
# pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
# print(pdf_page.page_content)

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE
)
split_docs = text_splitter.split_documents(texts)
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

# 通过智谱AIembddding 构建 向量库
embedding = ZhipuAIEmbeddings()
# 定义持久化路径
persist_directory = '../vector_chroma'


vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)
# vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")

# # 测试匹配，根据向量数据库  <<<相似度检索>>>
question = "核心能力有哪些"
sim_docs = vectordb.similarity_search(question, k=3)
print(f"检索到的内容")
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容：\n{sim_doc.page_content[:200]}", end="\n--------------------------------------\n")

 # 测试匹配，根据向量数据库  <<<MMR 检索>>>
# mmr_docs = vectordb.max_marginal_relevance_search(question, k=3)
# for i, sim_doc in enumerate(mmr_docs):
#     print(f"MMR检索到第{i}个内容：\n{sim_doc.page_content[:200]}", end="\n------------------------------------------------------------\n")



