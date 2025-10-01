---
title: YoutubeRAG
app_file: main.py
sdk: gradio
sdk_version: 5.46.0
---
# YouTube RAG System

> Live demo: [https://04f8fc1ccce8506c61.gradio.live](https://04f8fc1ccce8506c61.gradio.live)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)

_Languages: English · [中文](./README.zh-CN.md)_

A **lightweight**, **modular** YouTube video Q&A tool with **persistent session management**. Supports automatic subtitle extraction or audio transcription to build a searchable knowledge base.

<p align="center">
	<img src="./images/Demo.png" alt="Demo Screenshot" width="720" />
	<br/>
	<em>Web Interface: Interactive Q&A with session management</em>
</p>

## ✨ New in v2.0

- 🏗️ **Modular Architecture**: Clean, maintainable code structure
- 💾 **Session Persistence**: Save and reload RAG sessions
- 🌐 **Web Interface**: Beautiful Gradio-based UI
- 📱 **Simplified CLI**: Streamlined command-line experience
- 🔧 **Better Error Handling**: More robust and user-friendly

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Features

### Core Features
- 🎥 **Smart Content Extraction**: Automatic YouTube subtitle extraction with audio transcription fallback
- 📝 **AI Summarization**: Generate concise video summaries using OpenAI models
- 🔍 **Vector Search**: Intelligent Q&A using RAG (Retrieval Augmented Generation)
- 💾 **Session Persistence**: Save and reload analysis sessions
- 🌐 **Multi-language Support**: Chinese and English interfaces

### Interface Options
- 🖥️ **Web Interface**: User-friendly Gradio web UI
- ⌨️ **Command Line**: Streamlined CLI for developers
- 🔄 **Session Management**: List, load, save, and delete sessions

### Technical Features
- 🏗️ **Modular Design**: Clean separation of concerns
- 🤖 **Multiple AI Models**: Support for GPT-3.5, GPT-4, GPT-4o series
- 📊 **Chunked Processing**: Handle long videos efficiently
- 🛡️ **Error Resilience**: Robust error handling and recovery

## Installation

### Prerequisites
- Python 3.12+
- OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start

### 🌐 Web Interface (Recommended)

```bash
# Launch web interface (default mode)
python main.py

# Or explicitly specify UI mode
python main.py --ui
```

Open your browser and navigate to `http://localhost:7860`

### ⌨️ Command Line Interface

```bash
# Analyze a YouTube video
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Use different AI model
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --model gpt-4

# Skip Whisper transcription fallback (only use subtitles)
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --no-transcription

# Custom chunking parameters
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --chunk-size 1500 --chunk-overlap 50
```

## Usage

### Web Interface

1. **Start the application**: `python main.py`
2. **Enter API Key**: Provide your OpenAI API key
3. **Choose Action**:
   - **New Video**: Analyze a fresh YouTube video
   - **Load Session**: Continue with a previously saved session
4. **Ask Questions**: Interact with the AI about the video content

### Session Management

```bash
# List all saved sessions
python main.py --list-sessions

# Load a specific session
python main.py --load-session "session_name"

# Delete a session
python main.py --delete-session "session_name"

# Generate a detailed analysis report for a saved session
python main.py --export-analysis "session_name" --analysis-language zh

# Ingest every video from a YouTube playlist into one session
python scripts/playlist_ingest.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" \
  --session-name buffet_all --skip-existing --save-summary
```

### Advanced Usage

#### Custom Model Configuration

```bash
# Use GPT-4 for better quality (slower, more expensive)
python main.py --url "VIDEO_URL" --model gpt-4

# Use GPT-4o-mini for faster processing
python main.py --url "VIDEO_URL" --model gpt-4o-mini
```

#### Session Management Commands (in Web/CLI interface)

- `sessions` - View all saved sessions
- `save as [name]` - Save current session with custom name
- `reset` - Restart the application
- `exit` - Quit the application

## Project Structure

```
YouTube RAG System/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── README.zh-CN.md                # Chinese documentation
├── src/                           # Source code
│   └── youtube_rag_system/        # Main package
│       ├── __init__.py            # Package initialization
│       ├── core/                  # Core functionality
│       │   ├── __init__.py
│       │   ├── rag_engine.py      # Main RAG engine
│       │   ├── content_processor.py # Video processing
│       │   └── session_manager.py  # Session persistence
│       ├── ui/                    # User interfaces
│       │   ├── __init__.py
│       │   └── gradio_interface.py # Web interface
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── validators.py      # Input validation
│           └── file_utils.py      # File operations
├── rag_sessions/                  # Saved sessions (auto-created)
│   └── [session_id]/
│       ├── metadata.json         # Session metadata
│       └── chroma_db/            # Vector database
└── images/                       # Documentation images
    ├── Demo.png
    └── Demo2.png
```

## How It Works

1. **Content Acquisition**: 
   - First attempts to extract YouTube auto-generated subtitles
   - Falls back to audio download and Whisper transcription if needed

2. **AI Processing**:
   - Generates comprehensive video summary
   - Splits content into optimized chunks for vector search

3. **Knowledge Base Creation**:
   - Creates vector embeddings using OpenAI embeddings
   - Stores in ChromaDB for efficient retrieval

4. **Session Persistence**:
   - Saves all processed data to disk
   - Enables quick reload without reprocessing

5. **Interactive Q&A**:
   - Uses RAG to answer questions based on video content
   - Maintains context and provides relevant responses

## Command Line Options

```bash
python main.py [-h] [--ui] [--url URL] [--model MODEL] 
                   [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP] 
                   [--list-sessions] [--load-session SESSION] 
                   [--delete-session SESSION]

YouTube RAG System - YouTube视频RAG问答工具

optional arguments:
  -h, --help            show help message and exit
  --ui                  Launch Gradio web interface
  --url URL             YouTube video URL for CLI mode
  --model MODEL         OpenAI model name (default: gpt-3.5-turbo)
  --chunk-size SIZE     Text chunk size (default: 1000)
  --chunk-overlap SIZE  Text chunk overlap (default: 20)
  --list-sessions       List all saved sessions
  --load-session NAME   Load a saved session by name
  --delete-session NAME Delete a saved session by name
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Requirements

- Python 3.12+
- OpenAI API key
- Internet connection (for YouTube access and OpenAI API)

### Dependencies

See [requirements.txt](requirements.txt) for the complete list of Python packages.

## License

This project is licensed under the [MIT License](./LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
