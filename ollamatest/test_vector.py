"""
@author: VL
@time: 2024/9/3 9:24
@file: test_vector.py
@project: RAGandLangChain
@description: 读取单个文件并对其分割并测试其向量化使用的token
"""
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 知识库中单段文本长度
CHUNK_SIZE = 400
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 100

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../fileSource/2024低空经济行业研究报告-深企投产业研究院-2024.05-59页.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()
pdf_page = pdf_pages[15]

# 数据清晰之一数据清洗
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
print(pdf_page.page_content)

# 使用递归字符文本分割器中文本分割器都根据 chunk_size (块大小)和 chunk_overlap (块与块之间的重叠大小)进行分割。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
text_splitter.split_text(pdf_page.page_content[0:1000])
split_docs = text_splitter.split_documents(pdf_pages)
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")



















# print(f"每一个元素的类型：{type(pdf_page)}.",
#       f"该文档的描述性数据：{pdf_page.metadata}",
#       f"查看该文档的内容:\n{pdf_page.page_content}",
#       sep="\n------\n")
#
# print(f"载入后的变量类型为：{type(pdf_pages)}，", f"该 PDF 一共包含 {len(pdf_pages)} 页")
