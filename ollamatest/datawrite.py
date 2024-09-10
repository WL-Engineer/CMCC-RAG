"""
@author: VL
@time: 2024/9/3 9:24
@file: datawrite.py
@project: RAGandLangChain
@description: 持久化数据，与数据库连接函数
"""
import pymysql


def insertdata(llm, quetion, response):
    db = pymysql.connect(
        host='localhost',
        user='root',
        password='Vl100432588@',
        port=3306,
        charset='utf8mb4',
        database='pyllm'
    )
    try:
        cursor = db.cursor()
        sql = "INSERT INTO history (大模型型号, 请求问题, 回答) VALUES (%s, %s, %s)"
        cursor.execute(sql, (llm, quetion, response))
        db.commit()
    finally:
        cursor.close()
        db.close()


def vectorWrite(embedding, vectorText, response):
    db = pymysql.connect(
        host='localhost',
        user='root',
        password='Vl100432588@',
        port=3306,
        charset='utf8mb4',
        database='pyllm'
    )
    try:
        cursor = db.cursor()
        sql = "INSERT INTO embedding (模型名称, 向量词, 向量结果) VALUES (%s, %s, %s)"
        cursor.execute(sql, (embedding, vectorText, response))
        db.commit()
    finally:
        cursor.close()
        db.close()
