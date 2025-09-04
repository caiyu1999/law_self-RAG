import re 
import json
import time 
from typing import List
from typing_extensions import TypedDict, Literal

from config import Config
from make_collection import get_index

from IPython.display import Image, display
import streamlit as st
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_community.retrievers.llama_index import LlamaIndexRetriever
# from langchain_community.retrievers.llama_index import LlamaIndexGraphRetriever
from langchain.schema import Document
from pydantic import BaseModel, Field

from templates import (
                    prompt,
                    grade_prompt, 
                    hallucination_prompt, 
                    answer_prompt,
                    re_write_prompt,
                    GradeAnswer,
                    GradeDocuments,
                    GradeHallucinations)


st.set_page_config(
    page_title="智能劳动法咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

def disable_streamlit_watcher():
    """修补 Streamlit，禁用文件监视器"""
    def _on_script_changed(_):
        return
        
    from streamlit import runtime
    runtime.get_instance()._on_script_changed = _on_script_changed
    
    
# ========= 初始化程序 ==========
config = Config()
embed_model, llm, reranker,index = get_index(config)

retriever = index.as_retriever(similarity_top_k=Config.TOP_K,vector_store_query_mode="hybrid",alpha=0.5)



llm_doc_wso = llm.with_structured_output(GradeDocuments)
llm_answer_wso = llm.with_structured_output(GradeAnswer)
llm_hallucination_wso = llm.with_structured_output(GradeHallucinations)

llm_origin = prompt | llm  # 用于生成答案
llm_rewrite = re_write_prompt | llm  # 用于问题重写
llm_doc = grade_prompt | llm_doc_wso  # 用于文档相关性评分
llm_answer = answer_prompt | llm_answer_wso  # 用于判断是否解决问题
llm_hallucination = hallucination_prompt | llm_hallucination_wso  # 用于幻觉检测 

# ======== 定义 GraphState 和各节点函数 ==========
class GraphState(TypedDict):
    """
    表示图的状态。

    属性:
        question: 用户的问题。
        generation: LLM 生成的答案。
        documents: 检索到的文档列表。
    """
    question: str
    generation: str
    documents: List[Document]

def retrieve_node(state: GraphState)->GraphState:
    """
    检索与用户问题相关的法律条款，并进行重排序

    参数:
        state (GraphState): 当前图状态。

    返回:
        GraphState: 更新后的状态，包含检索到的文档。
    """
    question = state["question"]
    
    # 初级检索
    # print(type(question),question)
    initial_nodes = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=question)
  
    # 过滤低分，并将 nodes 转为 Document 
    documents = []
    for node in reranked_nodes:
        if node.score >= config.MIN_RERANK_SCORE:
            doc = Document(
                page_content=node.get_content(),
                metadata=node.metadata
            )
            documents.append(doc)
            
    return {"documents": documents, "question": question}
     
def generate_node(state: GraphState) -> GraphState:
    """
    使用 RAG 链生成答案。

    参数:
        state (GraphState): 当前图状态。

    返回:
        GraphState: 更新后的状态，包含生成的答案。
    """
    # print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # 使用 RAG 链生成答案
    generation = llm_origin.invoke({"context": documents, "question": question})
    # print(f"---GENERATION---\n{generation.content}\n")
    return {"documents": documents, "question": question, "generation": generation.content}

def grade_documents_node(state: GraphState) -> GraphState:
    """
    对检索到的文档与问题的相关性进行评分。

    参数:
        state (GraphState): 当前图状态。

    返回:
        GraphState: 更新后的状态，包含过滤后的相关文档。
    """
    # print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # 根据相关性过滤文档
    filtered_docs = []
    for d in documents:
        score = llm_doc.invoke(
            {"question": question, "document": d.metadata['full_title']+d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            # print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query_node(state: GraphState) -> GraphState:
    """
    将用户的问题重写为更适合检索的版本。

    参数:
        state (GraphState): 当前图状态。

    返回:
        GraphState: 更新后的状态，包含重写后的问题。
    """
    # print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # 重写问题以便更好检索
    better_question = llm_rewrite.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state: GraphState) -> Literal["transform_query_node", "generate_node"]:
    """
    决定是生成答案还是重写问题。

    参数:
        state (GraphState): 当前图状态。

    返回:
        Literal["transform_query_node", "generate_node"]: 下一个节点的决策。
    """
    # print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 如果没有相关文档，则重写问题
        # print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query_node"
    else:
        # 如果有相关文档，则生成答案
        # print("---DECISION: GENERATE---")
        return "generate_node"
     
def decide_generation_useful(state: GraphState) -> Literal["generate_node", "transform_query_node", END]:
    """
    判断生成的答案是否有用，或是否需要重新生成。

    参数:
        state (GraphState): 当前图状态。

    返回:
        Literal["generate_node", "transform_query_node", END]: 下一个节点的决策。
    """
    # print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 检查生成内容是否基于文档
    score = llm_hallucination.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 检查生成内容是否回答了问题
        # print("---GRADE GENERATION vs QUESTION---")
        score = llm_answer.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            # print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return END
        else:
            # print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "transform_query_node"
    else:
        # print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate_node"

# 辅助函数，将不可序列化对象转换为字典
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):              # 检查对象是否有 .dict() 方法
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # 处理列表和元组
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):           # 处理字典
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:                                 # 如果已可序列化则直接返回
        return obj

# ================== 界面组件 ==================
def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容
        
        with st.chat_message(role):
            st.markdown(content)
            
            # 如果是助手消息且包含思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                  unsafe_allow_html=True)
            
            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])

def show_reference_details(docs):
    with st.expander("查看支持依据"):
        for idx, documnet in enumerate(docs, 1):
            meta = documnet.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            # st.markdown(f"相关度：`{documnet.score:.4f}`")
            # st.info(f"{node.node.text[:300]}...")
            st.info(f"{documnet.page_content[:300]}...")
            
            
if __name__ == "__main__":
    disable_streamlit_watcher()
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统，请输入您的问题，我们将基于最新劳动法律法规为您解答。")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []
        # 构建图
        
    
    workflow = StateGraph(GraphState)

    # 定义节点
    workflow.add_node("retrieve_node", retrieve_node)                # 检索文档
    workflow.add_node("grade_documents_node", grade_documents_node)  # 文档相关性评分
    workflow.add_node("generate_node", generate_node)                # 生成答案
    workflow.add_node("transform_query_node", transform_query_node)  # 重写问题

    # 构建图的边
    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_documents_node")
    workflow.add_conditional_edges("grade_documents_node", decide_to_generate, ["transform_query_node", "generate_node"])
    workflow.add_edge("transform_query_node", "retrieve_node")
    workflow.add_conditional_edges("generate_node", decide_generation_useful, ["generate_node", "transform_query_node", END])

    # 编译工作流
    app = workflow.compile()
    # inputs = {"question": f"{prompt}"}
    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(key)
                        
    # # 可视化图（可选，需要额外依赖）
    # try:
    #     display(Image(app.get_graph(xray=True).draw_mermaid_png('./image.png')))
    # except Exception:
    #     pass
    
    init_chat_interface()
    
    if prompt := st.chat_input("请输入劳动法相关问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("正在分析问题..."):
            start_time = time.time()
            inputs = {"question": f"{prompt}"}
            # response_text = app.invoke(inputs)
            end_time = time.time()
            list_doc = []
            generate_list = []
            for output in app.stream(inputs):
                for key, value in output.items():
                    if "generation" in value:
                        generate_list.append(value['generation'])
                    if "documents" in value :
                        list_doc.append(value['documents'])
                    
            print(list_doc)
            print(generate_list)
            response_text = generate_list[-1] if generate_list else "抱歉，未能生成回答。" #获取最终回答
            filtered_docs = list_doc[-1] if list_doc else []  #获取最终回答所依赖的文档
            
            
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # 显示清理后的回答
            st.markdown(cleaned_response)
            
            # 如果有思维链内容则显示
            if think_contents:
                with st.expander("📝 模型思考过程（点击展开）"):
                    for content in think_contents:
                        st.markdown(f'<span style="color: #808080">{content.strip()}</span>', 
                                    unsafe_allow_html=True)
            
            # 显示参考依据（保持原有逻辑）
            show_reference_details(filtered_docs[:3])

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents  # 存储思维链内容
            })
