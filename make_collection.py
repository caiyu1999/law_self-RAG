import json 
from typing import List, Dict 
from pathlib import Path
from config import Config
import chromadb 
import streamlit as st 
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from langchain.chat_models import init_chat_model

@st.cache_resource(show_spinner="初始化模型中...")
def init_models(config: Config):
    """初始化模型并验证"""
    # Embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_PATH
    )
    reranker = SentenceTransformerRerank(
        model=config.RERANK_MODEL_PATH,
        top_n=config.RERANK_TOP_K
    )
    # llm = Ollama(
    #     model='qwen3:8b',
    # )
    llm = init_chat_model(
        model = 'gpt-4o-mini',
        base_url = config.BASE_URL,
        api_key = config.API_KEY,
        temperature = 0.3
    )
    
    Settings.embed_model = embed_model
    
    return embed_model, llm, reranker


@st.cache_resource(show_spinner="加载知识库中...")
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

@st.cache_resource(show_spinner="创建节点中...")
def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"
            
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)

    return nodes 



@st.cache_resource(show_spinner="初始化向量数据库...")
def init_vector_store(config:Config,nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=config.VECTOR_DB_DIR)
    
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 确保存储上下文正确初始化
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要新建索引
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")
        
        # 显式将节点添加到存储上下文
        storage_context.docstore.add_documents(nodes)  
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )

        # 双重持久化保障
        storage_context.persist(persist_dir=config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=config.PERSIST_DIR)  # <-- 新增
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证
    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")
    
    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")
    return index






@st.cache_resource(show_spinner="预备中...")
def get_index(config:Config) ->any:
    data_dir = config.DATA_DIR  
    embed_model, llm, reranker = init_models(config)
    all_data = load_and_validate_json_files(data_dir) 
    nodes = create_nodes(all_data)
    index = init_vector_store(config,nodes)
    
    return embed_model, llm, reranker,index 






if __name__ == "__main__":
    config = Config()
    print(config)
    embed_model, llm, reranker , index = get_index(config)
    
   
    

