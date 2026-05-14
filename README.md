# On-Call Assistant Search API

当前项目已合并 Phase 1 关键词检索和 Phase 2 语义搜索，基于 FastAPI 提供统一的 REST API 和两个独立搜索页面。

## 功能概览

- `POST /v1/documents`：写入 HTML 文档并同步更新关键词索引与语义索引
- `GET /v1/search?q=...`：Phase 1 关键词检索
- `GET /v2/search?q=...`：Phase 2 语义搜索，支持自然语言中文查询
- `GET /v1`：关键词搜索页面
- `GET /v2`：语义搜索页面
- 启动时自动加载 `data/` 目录中的 10 份 SOP 文档
- 使用 `BeautifulSoup` 移除 `script` 和 `style` 后提取正文
- 使用 `sentence-transformers` 中文友好向量模型做余弦相似度检索

## 项目结构

```text
onCall/
├── app.py
├── main.py
├── semantic_search.py
├── utils.py
├── requirements.txt
├── README.md
├── templates/
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
- 第一次安装和第一次模型下载耗时会明显更长

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
- 关键词接口：`http://127.0.0.1:8000/v1/search?q=OOM`
- 语义接口：`http://127.0.0.1:8000/v2/search?q=服务器挂了`
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

## 验证建议

### Phase 1

```bash
curl "http://127.0.0.1:8000/v1/search?q=OOM"
curl "http://127.0.0.1:8000/v1/search?q=故障"
curl "http://127.0.0.1:8000/v1/search?q=replication"
curl "http://127.0.0.1:8000/v1/search?q=CDN"
curl "http://127.0.0.1:8000/v1/search?q=%26"
```

预期：

- `OOM` 命中 `sop-001`
- `故障` 返回多个文档
- `replication` 返回空结果
- `CDN` 返回 `sop-003`、`sop-010`
- `%26` 返回正文包含 `&` 的文档

### Phase 2

```bash
curl "http://127.0.0.1:8000/v2/search?q=服务器挂了"
curl "http://127.0.0.1:8000/v2/search?q=黑客攻击"
curl "http://127.0.0.1:8000/v2/search?q=机器学习模型出问题"
```

预期目标：

- `服务器挂了`：`sop-001`、`sop-004` 靠前
- `黑客攻击`：`sop-005` 明显靠前
- `机器学习模型出问题`：`sop-008` 排在最前

## 后续扩展建议

当前结构已适合继续扩展：

- 在 `semantic_search.py` 中引入 chunk 级召回
- 在 Phase 3 中增加 RAG 检索和 LLM 生成回答
- 在语义召回后加入交叉编码器或 LLM 重排序
