# YouTube RAG 问答工具

一个轻量级的YouTube视频问答工具，支持自动提取字幕或音频转录，构建可搜索的知识库。

## 功能特点

- 🎥 自动获取YouTube视频字幕
- 🎙️ 备选音频转录（使用OpenAI Whisper）
- 🔍 向量化搜索和智能问答
- 💬 交互式命令行界面
- 🌐 支持中英文字幕

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置API密钥

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 运行工具

```bash
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## 使用示例

```bash
# 基本使用
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 自定义文本块大小
python3.12 youtube_rag.py "https://www.youtube.com/watch?v=iF9iV4xponk" --chunk-size 1500 --chunk-overlap 50
```

## 工作原理

1. **内容获取**: 优先尝试获取YouTube自动生成的字幕
2. **备选方案**: 如无字幕，则下载音频并使用Whisper转录
3. **文本处理**: 将内容分割成适合的文本块
4. **向量化**: 使用OpenAI embeddings创建向量数据库
5. **问答**: 基于检索增强生成(RAG)回答用户问题

## 依赖项

- Python 3.12
- OpenAI API密钥
- yt-dlp (YouTube视频下载)
- LangChain (RAG框架)
- ChromaDB (向量数据库)

## 许可证

MIT License