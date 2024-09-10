from __future__ import annotations
import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
"""
@time: 2024/9/3 9:24
@file: zhipuai_embedding.py
@project: RAGandLangChain
@description: langchain封装embedding
"""

"""
参考链接：https://github.com/datawhalechina/llm-universe/blob/4182b48827947a4d95453f88d3b01478a9548e39/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/%E9%99%84LangChain%E8%87%AA%E5%AE%9A%E4%B9%89Embedding%E5%B0%81%E8%A3%85%E8%AE%B2%E8%A7%A3.ipynb
"""

MODEL = "embedding-2"
logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    client: Any
    """`zhipuai.ZhipuAI"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
           实例化ZhipuAI为values["client"]
           Args:
               values (Dict): 包含配置信息的字典，必须包含 client 的字段.
           Returns:
               values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI()
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model=MODEL,
            input=text
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            生成输入文本列表的 embedding.
            Args:
                texts (List[str]): 要生成 embedding 的文本列表.

            Returns:
                List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
            """

        return [self.embed_query(text) for text in texts]
