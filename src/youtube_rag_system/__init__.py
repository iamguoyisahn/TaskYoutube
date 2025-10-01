"""
YouTube RAG System - A lightweight YouTube video Q&A tool
轻量级YouTube视频问答工具

Supports subtitle extraction and audio transcription to build a searchable knowledge base
支持字幕提取和音频转录，构建可搜索的知识库
"""

__version__ = "2.0.0"
__author__ = "YouTube RAG System"

from .core.rag_engine import YouTubeRAG
from .core.session_manager import SessionManager
from .ui.gradio_interface import YouTubeRAGInterface

__all__ = [
    'YouTubeRAG',
    'SessionManager', 
    'YouTubeRAGInterface'
]