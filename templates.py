from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate 

'''检索评分器'''
class GradeDocuments(BaseModel):
    """评估检索到的法律条款与用户问题的相关性"""

    binary_score: str = Field(description="检测到的法律条款是否与用户的问题相关?, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="生成的答案是否基于检索到的文档, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="生成的答案是否充分解决了用户的问题, 'yes' or 'no'")


system = '''你是一个劳动法智能助手，你需要基于检索到的法律条款来回答用户的问题。
如果检索到的法律条款能够支持你的回答，请务必引用相关法律条款，并在回答中注明“根据<法律名称>的第<条款编号>条款，……”。
如果检索到的法律条款无法支持你的回答，请明确告知'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户的问题: \n\n {question} \n\n 法律条款：{context}请基于检索到的法律条款作答。"),
    ]
)




# Define system prompt for grading relevance
system = """你是一名法律条款方面的评审员，负责评估检索到的法律条款与用户问题的相关性。\n
    评估不需要过于严格，目的是过滤掉明显错误的检索结果。\n
    如果法律条款包含与用户问题相关的关键词或语义内容，请判定为相关。\n
    请用“yes”或“no”二元分数表示该法律条款是否与用户问题相关。"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索得到的法律条款: \n\n {document} \n\n 用户的问题: {question}"),
    ]
)
'''Hallucination Grader 幻觉评分器'''


system = """你是一名法律条款方面的评审员，负责评估大模型生成的答案是否基于检索到的法律条款。\n
    如果答案内容与检索到的事实一致或得到支持，请判定为“yes”；否则为“no”。\n
    请用“yes”或“no”二元分数表示答案是否基于检索到的事实。"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的事实: \n\n {documents} \n\n 大模型生成的答案: {generation}"),
    ]
)


# Define system prompt for answer grading
system = """你是法律条款方面的一个评估答案是否解答/解决问题的评分器\n
给出一个二元评分‘是’或‘否’。‘是’表示该答案解决了问题。"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户的问题: \n\n {question} \n\n 大模型生成的答案: {generation}"),
    ]
)


''' question re-writer'''

system = """你是一个法律条款方面的问题重写器，负责将输入问题转化为更适合向量数据库检索的优化版本\n
通过分析输入内容，推理其潜在的语义意图/真实含义。"""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "初始问题: \n\n {question} \n 将其转换为更专业的检索问题",
        ),
    ]
)