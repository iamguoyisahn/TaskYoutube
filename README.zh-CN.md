# DeepYoutubeRAG（中文）

_语言： [English](./README.md) · **中文**_

DeepYoutubeRAG 是一个轻量级的 YouTube 视频问答（RAG）工具。它会尝试先获取视频自动生成的字幕；如果字幕不可用，则回退为下载音频并使用语音识别（例如 OpenAI Whisper）来生成转录文本。处理后将文本切分、向量化并存入向量数据库，从而支持基于检索的问答和语义搜索。

主要使用场景：
- 快速为单个或批量 YouTube 视频构建可搜索的知识库
- 在命令行环境中进行交互式问答或作为后端数据准备管道

## 主要功能

- 🎥 自动提取 YouTube 视频字幕（优先）
- 🎙️ 无字幕时回退到音频转写（Whisper 或其他 ASR）
- 🔍 文本分块、向量化（使用 OpenAI Embeddings）和向量检索（Chroma 等）
- 🧠 基于检索增强生成（RAG）的问答能力
- 💬 命令行交互支持中文和英文

## 快速开始（5 分钟上手）

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/iamguoyisahn/TaskYoutube.git
cd TaskYoutube
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 配置环境变量（至少需要 OpenAI API Key）：

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

5. 运行一次示例：

```bash
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

替换 `VIDEO_ID` 为实际的视频 ID。例如：`dQw4w9WgXcQ`。

## 使用示例（常见选项）

```bash
# 基本使用
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 自定义分块大小和重叠
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --chunk-size 1500 --chunk-overlap 50
```

## 工作流程（内部步骤）

1. 内容获取：优先尝试获取 YouTube 自动字幕（或手动上传的字幕）。
2. 回退转录：若无字幕，使用 `yt-dlp` 下载音轨并通过 Whisper（或其他 ASR）转录为文本。
3. 文本清洗与分块：对转录文本进行基本清洗并按配置的 `chunk_size`/`overlap` 切分。
4. 向量化：使用 OpenAI Embeddings 将每个文本片段转为向量，并写入向量数据库（默认 Chroma，可扩展）。
5. 检索与生成：在用户查询时，先检索相关片段再通过 LLM（可配置）生成回答。

## 依赖与要求

- Python 3.12
- `OPENAI_API_KEY`（必需，用于 Embeddings / LLM）
- `yt-dlp`（用于下载视频/音频/字幕）
- `whisper` 或其它 ASR（在需要转录时）
- `langchain`、`chromadb` 等库（见 `requirements.txt`）

## 命令行参数

```text
python3.12 youtube_rag.py [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP] url

positional arguments:
  url                   YouTube video URL

optional arguments:
  -h, --help            show this help message and exit
  --chunk-size CHUNK_SIZE
                        Text chunk size (default: 1000)
  --chunk-overlap CHUNK_OVERLAP
                        Text chunk overlap (default: 20)
```

## 环境变量

- `OPENAI_API_KEY`：OpenAI API Key（必需）
- 可选：`WHISPER_MODEL`、`CHROMA_DIR` 等（如在脚本中暴露）

## 项目结构（简要）

```
TaskYoutube/
├── youtube_rag/
│   └── youtube_rag.py   # 主脚本/入口
├── requirements.txt
├── README.md
├── README.zh-CN.md
└── LICENSE
```

## 贡献

欢迎贡献！常见流程：fork → 新建分支 → 提交 PR。请在 PR 描述中包含复现步骤与简要说明。

## 许可证

本项目采用 [MIT License](./LICENSE)。
