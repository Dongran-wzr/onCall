# On-Call Assistant Search API

Phase 1 实现了一个基于 FastAPI 的文档搜索引擎 API，用于加载 `data/` 目录中的部门 On-Call SOP HTML 文档，并提供关键词检索能力。

## 功能特性

- `POST /v1/documents`：接收 HTML 文档，使用 BeautifulSoup 提取标题并清洗正文
- `GET /v1/search?q=...`：基于 `clean_text` 做大小写不敏感搜索，支持中文、英文和特殊字符
- `GET /v1`：提供一个简单的搜索页面，使用内联 CSS 和少量 JavaScript 实现实时搜索
- 启动时自动加载 `data/` 目录下所有 `*.html` 文件，文件名作为文档 ID
- 移除 `script` 和 `style` 内容，避免干扰检索
- 添加了 CORS 支持，方便前端页面或其他客户端调用

## 项目结构

```text
onCall/
├── app.py
├── main.py
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

## 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 启动服务

在项目根目录执行：

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

或直接运行：

```bash
python main.py
```

服务启动后可访问：

- 搜索页面：`http://127.0.0.1:8000/v1`
- 搜索接口：`http://127.0.0.1:8000/v1/search?q=OOM`
- 健康检查：`http://127.0.0.1:8000/health`

## API 示例

### 1. 新增文档

```bash
curl -X POST "http://127.0.0.1:8000/v1/documents" ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"sop-999\",\"html\":\"<html><head><title>测试文档</title></head><body><h1>测试文档</h1><p>这里包含故障、OOM 和 CDN 关键字。</p></body></html>\"}"
```

预期返回：

```json
{
  "id": "sop-999",
  "title": "测试文档"
}
```

### 2. 搜索文档

```bash
curl "http://127.0.0.1:8000/v1/search?q=OOM"
```

搜索 `&` 时请对查询参数做 URL 编码：

```bash
curl "http://127.0.0.1:8000/v1/search?q=%26"
```

## 关键实现说明

- 标题提取优先使用 `<title>`，如果不存在则回退到首个 `<h1>`，再回退到 `Document {id}`
- 正文清洗使用 `BeautifulSoup`，移除所有 `<script>` 和 `<style>` 标签后提取可见文本
- 搜索采用简单 TF 思路，分数为关键词在 `clean_text` 中出现的次数
- snippet 会截取首次命中的上下文片段，长度约 100-150 字符
- 空查询会返回全部文档，便于前端初始展示

## 建议验证

启动服务后，重点验证以下查询：

```bash
curl "http://127.0.0.1:8000/v1/search?q=OOM"
curl "http://127.0.0.1:8000/v1/search?q=故障"
curl "http://127.0.0.1:8000/v1/search?q=replication"
curl "http://127.0.0.1:8000/v1/search?q=CDN"
curl "http://127.0.0.1:8000/v1/search?q=%26"
```

预期：

- `OOM` 返回 `sop-001`
- `故障` 返回多个文档
- `replication` 返回空结果
- `CDN` 返回 `sop-003`、`sop-010` 等
- `%26` 返回正文包含 `&` 的文档
