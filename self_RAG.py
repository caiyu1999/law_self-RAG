from make_collection import get_index 
from config import Config
import pydantic 
import json 
from IPython.display import Image, display

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
from typing import List
from typing_extensions import TypedDict,Literal
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


config = Config()
embed_model, llm, reranker,index = get_index(config)

retriever = index.as_retriever(similarity_top_k=Config.TOP_K,vector_store_query_mode="hybrid",alpha=0.5)



llm_doc_wso = llm.with_structured_output(GradeDocuments)
llm_answer_wso = llm.with_structured_output(GradeAnswer)
llm_hallucination_wso = llm.with_structured_output(GradeHallucinations)

llm_origin = prompt | llm  # for generation
llm_rewrite = re_write_prompt | llm  # for query rewrite
llm_doc = grade_prompt | llm_doc_wso  # for document grading
llm_answer = answer_prompt | llm_answer_wso  # for 是否解决了问题
llm_hallucination = hallucination_prompt | llm_hallucination_wso  # for 幻觉检测 


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated answer.
        documents: List of retrieved documents.
    """
    question: str
    generation: str
    documents: List[Document]



def retrieve_node(state: GraphState)->GraphState:
    """
    检索与用户相关的问题的法律条款 并进行rerank

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with retrieved documents.
    """
    
    question = state["question"]
    
    # 初级检索
    initial_nodes = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=question)
  
    # 过滤低分 并将nodes转为Document 
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
    Generate an answer using the RAG chain.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with generated answer.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Generate answer using the RAG chain
    generation = llm_origin.invoke({"context": documents, "question": question})
    print(f"---GENERATION---\n{generation.content}\n")
    return {"documents": documents, "question": question, "generation": generation.content}


def grade_documents_node(state: GraphState) -> GraphState:
    """
    Grade the relevance of retrieved documents to the question.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with filtered relevant documents.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Filter documents based on relevance
    filtered_docs = []
    for d in documents:
        score = llm_doc.invoke(
            {"question": question, "document": d.metadata['full_title']+d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query_node(state: GraphState) -> GraphState:
    """
    Transform the user's question into a better version for retrieval.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with a rephrased question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Rewrite the question for better retrieval
    better_question = llm_rewrite.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state: GraphState) -> Literal["transform_query_node", "generate_node"]:
    """
    Decide whether to generate an answer or rephrase the question.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Literal["transform_query_node", "generate_node"]: Decision for the next node.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # If no relevant documents, rephrase the question
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query_node"
    else:
        # If relevant documents exist, generate an answer
        print("---DECISION: GENERATE---")
        return "generate_node"
    
    
def decide_generation_useful(state: GraphState) -> Literal["generate_node", "transform_query_node", END]:
    """
    Decide whether the generated answer is useful or needs to be regenerated.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Literal["generate_node", "transform_query_node", END]: Decision for the next node.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if the generation is grounded in the documents
    score = llm_hallucination.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check if the generation addresses the question
        print("---GRADE GENERATION vs QUESTION---")
        score = llm_answer.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return END
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "transform_query_node"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate_node"


# Helper function to convert non-serializable objects to dictionaries
def convert_to_serializable(obj):
    if hasattr(obj, "dict"):              # Check if the object has a .dict() method
        return obj.dict()
    elif isinstance(obj, (list, tuple)):  # Handle lists and tuples
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):           # Handle dictionaries
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:                                 # Return the object as-is if it's already serializable
        return obj





if __name__ == "__main__":
        # Build Graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve_node", retrieve_node)                # Retrieve documents
    workflow.add_node("grade_documents_node", grade_documents_node)  # Grade document relevance
    workflow.add_node("generate_node", generate_node)                # Generate answer
    workflow.add_node("transform_query_node", transform_query_node)  # Transform query

    # Build graph edges
    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_documents_node")
    workflow.add_conditional_edges("grade_documents_node", decide_to_generate, ["transform_query_node", "generate_node"])
    workflow.add_edge("transform_query_node", "retrieve_node")
    workflow.add_conditional_edges("generate_node", decide_generation_useful, ["generate_node", "transform_query_node", END])

    # Compile the workflow
    app = workflow.compile()

    # Visualize the graph (optional, requires additional dependencies)
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png('./image.png')))
    except Exception:
        pass
    
    
        # Example 1: Question about agent memory
    inputs = {"question": "为了保证劳动者的合法权益，工会应当做些什么?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("-"*80)
            print(value)
            # Convert non-serializable objects to dictionaries
            # serializable_value = convert_to_serializable(value)
            # Print the serialized value as a JSON string
            # print(json.dumps(serializable_value, indent=2))
        # print("="*80)

    # Print the final generation
    print(f"Final Generation:\n{value['generation']}\n")
