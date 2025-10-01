"""
File utilities for YouTube RAG System
YouTube RAG系统文件工具
"""

from pathlib import Path
from typing import Optional


def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL
    从YouTube URL提取视频ID
    
    Args:
        url: YouTube URL
        
    Returns:
        str: Video ID or 'unknown'
    """
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return "unknown"


def save_text_file(content: str, filename: str, header: str = "") -> Optional[str]:
    """
    Save text content to file
    保存文本内容到文件
    
    Args:
        content: Text content to save
        filename: Target filename
        header: Optional header text
        
    Returns:
        str: Filename if successful, None if failed
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            if header:
                f.write(f"{header}\n")
                f.write("=" * 50 + "\n\n")
            f.write(content)
        return filename
    except Exception:
        return None