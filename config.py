from dataclasses import dataclass
from typing import Optional


@dataclass 
class Config:
    '''配置类'''
    BASE_URL:str = "https://api.chatanywhere.tech/v1"
    API_KEY:str = "sk-Maf9m5KxsypZQ76kF2qQ6lsqLs3PL0cm2Bs3XeOD1yl6Lk86"
    
    EMBED_MODEL_PATH:str = "/home/tianfang/Agent/self-RAG/models/sungw111/text2vec-base-chinese-sentence"
    RERANK_MODEL_PATH:str = "/home/tianfang/Agent/self-RAG/models/BAAI/bge-reranker-large"
    
    DATA_DIR:str = "./data"
    VECTOR_DB_DIR:str = "./chroma_db"
    PERSIST_DIR:str = "./storage"
    
    COLLECTION_NAME:str = "chinese_labor_laws"
    TOP_K:int = 10
    RERANK_TOP_K:int = 5
    MIN_RERANK_SCORE = 0.4





if __name__ == "__main__":
    config = Config()
    print(config)