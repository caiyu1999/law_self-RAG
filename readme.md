# ⚖️ 智能劳动法咨询助手

基于Self-RAG（Self-Retrieval-Augmented Generation）技术的智能劳动法咨询系统，能够基于最新的劳动法律法规为用户提供准确、可靠的法律咨询服务。

## 🚀 项目特色

- **Self-RAG架构**: 采用先进的Self-RAG技术，通过自我检索、自我评估和自我改进机制，确保回答的准确性和可靠性
- **专业法律知识库**: 基于《中华人民共和国劳动合同法》等权威法律文件构建的知识库
- **智能问答系统**: 支持自然语言交互，能够理解复杂的劳动法问题并提供专业解答
- **实时参考依据**: 每个回答都会提供相关的法律条款作为支持依据
- **Web界面**: 基于Streamlit构建的现代化Web界面，操作简单直观

## 🏗️ 系统架构

### 核心组件

1. **检索模块** (`retrieve_node`)
   - 使用向量数据库进行语义检索
   - 支持混合检索模式（向量+关键词）
   - 集成重排序机制提升检索质量

2. **文档评估模块** (`grade_documents_node`)
   - 评估检索文档与问题的相关性
   - 过滤不相关的文档，提高回答质量

3. **答案生成模块** (`generate_node`)
   - 基于检索到的法律条款生成专业回答
   - 自动引用相关法律条款

4. **问题重写模块** (`transform_query_node`)
   - 优化用户问题以提升检索效果
   - 处理复杂或模糊的法律问题

5. **质量评估模块** (`decide_generation_useful`)
   - 检测回答是否基于检索文档（防幻觉）
   - 评估回答是否充分解决了用户问题

### 工作流程

```
用户问题 → 检索文档 → 文档相关性评估 → 生成答案 → 质量评估 → 输出结果
                ↓              ↓              ↓
            问题重写 ← 答案不充分 ← 检测到幻觉
```

## 📋 环境要求

- Python 3.8+
- CUDA支持（推荐，用于模型推理）
- 至少8GB内存
- 至少10GB磁盘空间

## 🛠️ 安装部署

### 1. 克隆项目

```bash
git clone <repository-url>
cd self-RAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `streamlit` - Web界面框架
- `langchain` - LLM应用框架
- `langchain-openai` - OpenAI集成
- `langchain-community` - 社区组件
- `langgraph` - 图工作流
- `llama-index` - 向量数据库和检索
- `chromadb` - 向量数据库
- `sentence-transformers` - 文本嵌入模型
- `pydantic` - 数据验证

### 3. 配置模型
在modelscope上下载rerank模型以及embedding模型

在 `config.py` 中配置以下参数：

```python
@dataclass 
class Config:
    BASE_URL: str = "your_openai_api_base_url"
    API_KEY: str = "your_openai_api_key"
    
    # 模型路径配置
    EMBED_MODEL_PATH: str = "path/to/embedding/model"
    RERANK_MODEL_PATH: str = "path/to/reranker/model"
    
    # 数据路径配置
    DATA_DIR: str = "./data"
    VECTOR_DB_DIR: str = "./chroma_db"
    PERSIST_DIR: str = "./storage"
    
    # 检索参数
    TOP_K: int = 10
    RERANK_TOP_K: int = 5
    MIN_RERANK_SCORE = 0.4
```

### 4. 准备数据

将劳动法相关文档放入 `data/` 目录，支持JSON格式：

```json
[
    {
        "中华人民共和国劳动合同法 第一条": "法律条款内容...",
        "中华人民共和国劳动合同法 第二条": "法律条款内容..."
    }
]
```

### 5. 运行应用

```bash
streamlit run self_RAG.py
```

访问 `http://localhost:8501` 即可使用系统。

## 📚 知识库内容

当前系统包含以下法律文件：

- **中华人民共和国劳动合同法** - 完整的98条法律条款
  - 劳动合同的订立、履行、变更、解除和终止
  - 试用期、服务期、竞业限制等特殊条款
  - 劳务派遣、非全日制用工等特殊用工形式
  - 法律责任和监督检查

## 🎯 使用示例

### 问题类型

系统可以回答以下类型的劳动法问题：

1. **劳动合同相关问题**
   - "试用期最长可以约定多长时间？"
   - "什么情况下可以解除劳动合同？"

2. **工资报酬问题**
   - "加班费如何计算？"
   - "最低工资标准是什么？"

3. **社会保险问题**
   - "用人单位必须为员工缴纳哪些保险？"

4. **劳动保护问题**
   - "劳动者有哪些权利？"
   - "工伤认定标准是什么？"

### 回答特点

- **准确性**: 基于权威法律条款，确保回答的准确性
- **可追溯性**: 每个回答都提供具体的法律条款依据
- **专业性**: 使用专业的法律术语和表达
- **实用性**: 针对实际问题提供实用的法律建议

## 🔧 技术细节

### 模型配置

- **嵌入模型**: text2vec-base-chinese-sentence
- **重排序模型**: bge-reranker-large
- **大语言模型**: GPT-4o-mini（可配置）

### 检索策略

- **混合检索**: 结合向量相似度和关键词匹配
- **重排序**: 使用专门的reranker模型提升检索质量
- **相关性过滤**: 通过LLM评估文档相关性

### 质量保证

- **幻觉检测**: 确保回答基于检索到的法律条款
- **问题解决评估**: 验证回答是否充分解决了用户问题
- **多轮优化**: 通过问题重写和重新生成提升回答质量


**免责声明**: 本系统提供的法律建议仅供参考，不构成正式的法律意见。具体法律问题请咨询专业律师。
