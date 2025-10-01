"""
Validation utilities for YouTube RAG System
YouTube RAG系统验证工具
"""

import re


def validate_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format
    验证OpenAI API密钥格式
    
    Args:
        api_key: API key string
        
    Returns:
        bool: True if valid format
    """
    if not api_key:
        return False
    return api_key.startswith('sk-') and len(api_key) > 20


def validate_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL format
    验证YouTube URL格式
    
    Args:
        url: YouTube URL string
        
    Returns:
        bool: True if valid YouTube URL
    """
    if not url:
        return False
    
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/v/[\w-]+'
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)