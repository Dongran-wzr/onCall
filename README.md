# On-Call Assistant Search API

当前项目已合并 Phase 1 关键词检索、Phase 2 语义搜索和 Phase 3 On-Call Agent，基于 FastAPI 提供统一的 REST API 和三个独立页面。

## 功能概览

- `POST /v1/documents`：写入 HTML 文档并同步更新关键词索引与语义索引
- `GET /v1/search?q=...`：Phase 1 关键词检索
- `GET /v2/search?q=...`：Phase 2 语义搜索，支持自然语言中文查询
- `POST /v3/chat`：Phase 3 Agent 对话接口，使用 OpenAI function calling + 受限 `readFile` 工具
- `GET /v1`：关键词搜索页面
- `GET /v2`：语义搜索页面
- `GET /v3`：Agent 对话页面，展示 reasoning、工具调用和最终回答
- 启动时自动加载 `data/` 目录中的 10 份 SOP 文档
- 使用 `BeautifulSoup` 移除 `script` 和 `style` 后提取正文
- 使用 `sentence-transformers` 中文友好向量模型做余弦相似度检索
- 使用 OpenAI Chat Completions 的 function calling 驱动 Agent

## 项目结构

```text
onCall/
├── agent.py
├── app.py
├── main.py
├── semantic_search.py
├── utils.py
├── requirements.txt
├── README.md
├── templates/
│   ├── chat.html
│   └── index.html
└── data/
    ├── sop-001.html
    ├── ...
    └── sop-010.html
```

## 语义搜索实现说明

Phase 2 使用 `BAAI/bge-small-zh-v1.5` 作为默认向量模型，原因如下：

- 中文检索效果通常优于通用英文小模型
- 模型体积适中，适合当前 10 份文档的小规模场景
- 可直接用于后续 Phase 3 的 RAG 检索或重排序扩展

实现细节：

- 应用启动时加载模型，并对每个文档的 `title + clean_text` 生成文档级 embedding
- 同时对文档段落生成 segment embedding，用于返回更相关的 snippet
- 查询时先对自然语言 query 生成 embedding，再与文档向量做余弦相似度计算
- 返回 `score > 0.3` 的结果，并按分数降序排序，最多返回 Top 10
- 对少量口语化 on-call 查询做了轻量同义扩展，例如“服务器挂了”“内存爆炸”“模型漂移”

性能说明：

- 当前仅有 10 份文档，embedding 在启动时预计算并缓存到内存
- 正常情况下，模型完成首次加载后，单次查询通常可以保持在 300ms 以内
- 第一次启动会自动从 Hugging Face 下载模型，后续会读取本地缓存

## Agent 实现说明

Phase 3 使用 `OpenAI function calling` 实现一个受限的 On-Call Agent，核心文件是 `agent.py`。

设计原则：

- Agent 只能调用一个工具：`readFile`
- 工具参数中的 `fname` 必须是精确文件名，例如 `sop-002.html`
- 禁止使用目录遍历、通配符、列目录、猜测路径
- Agent 必须先读取真实文件，再基于文件内容回答
- 页面和流式接口会展示 reasoning、工具调用、工具结果和最终回答

`readFile` 的行为：

- 当文件存在时，读取 `data/` 目录下的真实内容
- 当文件不存在且提供了 `content` 时，创建该文件
- 创建新的 `.html` 文件后，会自动同步到当前内存文档存储和语义索引

Prompt 设计重点：

- System Prompt 显式要求“先思考，再读文件，再回答”
- 预先提供 SOP 文件名和标题清单，避免 Agent 通过列目录寻找文件
- 同时提供基于现有检索能力推断的候选文件，帮助 Agent 更快选择正确 SOP

流式事件类型：

- `status`
- `retrieval`
- `thought`
- `tool_call`
- `tool_result`
- `final`

## 安装依赖

建议使用虚拟环境：

```bash
py -m venv .venv
.venv\Scripts\activate
.\.venv\Scripts\python -m pip install -r requirements.txt
```

说明：

- `sentence-transformers` 会依赖 `transformers` 等组件
- `torch` 用于本地向量编码计算
- `openai` 用于 Phase 3 Agent 的 function calling
- 第一次安装和第一次模型下载耗时会明显更长

## 环境变量

Phase 3 支持 OpenAI 兼容接口，也支持直接使用 Kimi/Moonshot 和 DeepSeek。

### 方案一：OpenAI

```bash
set OPENAI_API_KEY=你的密钥
set OPENAI_MODEL=gpt-4o-mini
```

### 方案二：Kimi / Moonshot

推荐直接这样配置：

```bash
set LLM_PROVIDER=kimi
set MOONSHOT_API_KEY=你的密钥
set KIMI_MODEL=moonshot-v1-8k
```

也可以不设置 `LLM_PROVIDER`，只设置 `MOONSHOT_API_KEY`，系统会自动切到 Kimi 的兼容接口。

### 方案三：DeepSeek

推荐直接这样配置：

```bash
set LLM_PROVIDER=deepseek
set DEEPSEEK_API_KEY=你的密钥
set DEEPSEEK_MODEL=deepseek-chat
```

也可以只设置 `DEEPSEEK_API_KEY`，系统会自动切到 DeepSeek 的兼容接口。

### 自定义兼容网关

如果你使用兼容 OpenAI API 的代理或网关，也可以设置：

```bash
set OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

## 启动服务

在项目根目录执行：

```bash
.\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

或直接运行：

```bash
.\.venv\Scripts\python main.py
```

服务启动后可访问：

- 关键词搜索页：`http://127.0.0.1:8000/v1`
- 语义搜索页：`http://127.0.0.1:8000/v2`
- Agent 对话页：`http://127.0.0.1:8000/v3`
- 关键词接口：`http://127.0.0.1:8000/v1/search?q=OOM`
- 语义接口：`http://127.0.0.1:8000/v2/search?q=服务器挂了`
- Agent 接口：`http://127.0.0.1:8000/v3/chat`
- 健康检查：`http://127.0.0.1:8000/health`

## API 示例

### 1. 新增文档

```bash
curl -X POST "http://127.0.0.1:8000/v1/documents" ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"sop-999\",\"html\":\"<html><head><title>测试文档</title></head><body><h1>测试文档</h1><p>这里包含服务宕机、故障恢复和模型异常排查说明。</p></body></html>\"}"
```

### 2. Phase 1 关键词搜索

```bash
curl "http://127.0.0.1:8000/v1/search?q=OOM"
curl "http://127.0.0.1:8000/v1/search?q=%26"
```

### 3. Phase 2 语义搜索

```bash
curl "http://127.0.0.1:8000/v2/search?q=服务器挂了"
curl "http://127.0.0.1:8000/v2/search?q=黑客攻击"
curl "http://127.0.0.1:8000/v2/search?q=机器学习模型出问题"
curl "http://127.0.0.1:8000/v2/search?q=内存爆炸"
curl "http://127.0.0.1:8000/v2/search?q=流量洪峰"
curl "http://127.0.0.1:8000/v2/search?q=模型漂移"
```

### 4. Phase 3 Agent

```bash
curl -X POST "http://127.0.0.1:8000/v3/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"数据库主从延迟超过30秒怎么处理？\",\"history\":[],\"stream\":false}"
```

预期返回：

```json
{
  "answer": "...",
  "events": [
    {"type": "status", "payload": {...}},
    {"type": "retrieval", "payload": {...}},
    {"type": "thought", "payload": {...}},
    {"type": "tool_call", "payload": {...}},
    {"type": "tool_result", "payload": {...}},
    {"type": "final", "payload": {...}}
  ]
}
```
