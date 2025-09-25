# YouTube RAG Q&A Tool Now Published Here (https://ba6211542da1631c92.gradio.live)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

_Languages: English · [中文](./README.zh-CN.md)_

A lightweight YouTube video Q&A tool that supports automatic subtitle extraction or audio transcription to build a searchable knowledge base.

<p align="center">
	<img src="./images/Demo.png" alt="Demo Screenshot" width="720" />
	<br/>
	<em>Demo: interactive Q&amp;A session</em>
</p>
## Table of Contents

- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Environment Variables](#environment-variables)
- [License](#license)

## Features

- 🎥 Automatic YouTube video subtitle extraction
- 🎙️ Fallback audio transcription (using OpenAI Whisper)
- 🔍 Vector search and intelligent Q&A
- 💬 Interactive command-line interface
- 🌐 Support for Chinese and English subtitles

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Tool

```bash
python3.12 youtube_rag/youtube_rag.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Usage Examples


```bash
# Basic usage
python3.12 youtube_rag/youtube_rag.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Custom chunk size
python3.12 youtube_rag/youtube_rag.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --chunk-size 1500 --chunk-overlap 50
```

Note: When running the tool, it may prompt you to confirm audio transcription (if subtitles are unavailable) and whether to save the original extracted/transcribed text to a `.txt` file. Follow the prompts in the terminal.

## How It Works

1. **Content Acquisition**: First attempts to get YouTube auto-generated subtitles
2. **Fallback Method**: If no subtitles available, downloads audio and uses Whisper for transcription
3. **Text Processing**: Splits content into appropriate text chunks
4. **Vectorization**: Creates vector database using OpenAI embeddings
5. **Q&A**: Answers user questions based on Retrieval Augmented Generation (RAG)



## Requirements

- Python 3.12
- OpenAI API key
- yt-dlp (YouTube video downloader)
- LangChain (RAG framework)
- ChromaDB (vector database)

## Command Line Options

```bash
python3.12 youtube_rag/youtube_rag.py [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP] url

positional arguments:
  url                   YouTube video URL

optional arguments:
  -h, --help            show this help message and exit
  --chunk-size CHUNK_SIZE
                        Text chunk size (default: 1000)
  --chunk-overlap CHUNK_OVERLAP
                        Text chunk overlap (default: 20)
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## License

This project is licensed under the [MIT License](./LICENSE).

