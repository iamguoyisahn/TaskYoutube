#!/usr/bin/env python3
"""
YouTube RAG - A lightweight YouTube video Q&A tool / 轻量级YouTube视频问答工具
Supports subtitle extraction and audio transcription to build a searchable knowledge base
支持字幕提取和音频转录，构建可搜索的知识库

Usage / 使用方法:
    python youtube_rag.py "https://www.youtube.com/watch?v=VIDEO_ID"
    
Environment Variables / 环境变量:
    OPENAI_API_KEY: OpenAI API key / OpenAI API密钥
"""

import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path

from langchain.schema import Document
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


class YouTubeRAG:
    """YouTube Video RAG Q&A System / YouTube视频RAG问答系统"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._check_openai_key()
        
    def _check_openai_key(self):
        """Check OpenAI API key / 检查OpenAI API密钥"""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Please set environment variable OPENAI_API_KEY / 请设置环境变量 OPENAI_API_KEY")
            sys.exit(1)
    
    def get_video_content(self, url):
        """Get video content, prioritize subtitles, fallback to audio transcription / 获取视频内容，优先字幕，备选音频转录"""
        print(f"🎥 Processing video / 处理视频: {url}")
        
        # Try to get subtitles / 尝试获取字幕
        content = self._get_subtitles(url)
        if content:
            print("✅ Using video subtitles / 使用视频字幕")
            return Document(page_content=content, metadata={"source": url, "type": "subtitles"})
        
        # Fallback: audio transcription / 备选：音频转录
        print("⚠️ No subtitles found, starting audio transcription... / 未找到字幕，开始音频转录...")
        content = self._transcribe_audio(url)
        print("✅ Using audio transcription / 使用音频转录")
        return Document(page_content=content, metadata={"source": url, "type": "transcription"})
    
    def _get_subtitles(self, url):
        """Get YouTube subtitles / 获取YouTube字幕"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                subtitle_path = Path(tmpdir) / "subtitle.%(ext)s"
                
                cmd = [
                    "python3.12", "-m", "yt_dlp",
                    "--write-subs", "--write-auto-subs",
                    "--sub-langs", "zh,zh-CN,en",
                    "--skip-download",
                    "-o", str(subtitle_path),
                    url
                ]
                
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Find subtitle files / 查找字幕文件
                for file in Path(tmpdir).glob("*.{vtt,srt}"):
                    return file.read_text(encoding='utf-8')
                    
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"Subtitle extraction failed / 字幕获取失败: {e}")
        return None
    
    def _transcribe_audio(self, url):
        """Download audio and transcribe / 下载音频并转录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.m4a"
            
            # Download audio / 下载音频
            cmd = [
                "python3.12", "-m", "yt_dlp",
                "-f", "bestaudio[ext=m4a]",
                "-o", str(audio_path),
                url
            ]
            subprocess.run(cmd, check=True)
            
            # Whisper transcription / Whisper转录
            client = OpenAI()
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            
            return transcript.text
    
    def build_knowledge_base(self, documents):
        """Build vector knowledge base / 构建向量知识库"""
        print("🔧 Building knowledge base... / 构建知识库...")
        
        # Text splitting / 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Vectorization and storage / 向量化存储
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        print(f"✅ Knowledge base built successfully / 知识库构建完成, {len(chunks)} chunks total / 共{len(chunks)}个片段")
        return vector_store.as_retriever()
    
    def create_qa_chain(self, retriever):
        """Create Q&A chain / 创建问答链"""
        system_prompt = """Answer user questions based on the following context.
If you don't know the answer, say "I don't know" and don't make up answers.
Please answer in the same language as the question.

基于以下上下文回答用户问题。
如果不知道答案，请说"我不知道"，不要编造答案。
请用与问题相同的语言回答。

Context / 上下文: {context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        llm = ChatOpenAI(temperature=0.1, max_tokens=2048)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        return create_retrieval_chain(retriever, question_answer_chain)
    
    def interactive_qa(self, rag_chain):
        """Interactive Q&A / 交互式问答"""
        print("\n🤖 Q&A system started! Type 'quit' to exit / 问答系统已启动！输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ Ask a question / 请提问: ").strip()
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not question:
                    continue
                
                print("🤔 Thinking... / 思考中...")
                result = rag_chain.invoke({"input": question})
                print(f"\n💡 {result['answer']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error occurred / 出错了: {e}")
        
        print("\n👋 Goodbye! / 再见！")


def main():
    parser = argparse.ArgumentParser(description="YouTube Video RAG Q&A Tool / YouTube视频RAG问答工具")
    parser.add_argument("url", help="YouTube video URL / YouTube视频URL")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Text chunk size / 文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Text chunk overlap / 文本块重叠")
    
    args = parser.parse_args()
    
    # Create RAG system / 创建RAG系统
    rag = YouTubeRAG(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    try:
        # Get video content / 获取视频内容
        document = rag.get_video_content(args.url)
        
        # Build knowledge base / 构建知识库
        retriever = rag.build_knowledge_base([document])
        
        # Create Q&A chain / 创建问答链
        qa_chain = rag.create_qa_chain(retriever)
        
        # Start interactive Q&A / 启动交互式问答
        rag.interactive_qa(qa_chain)
        
    except Exception as e:
        print(f"❌ Execution failed / 运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()