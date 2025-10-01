"""
Session management for YouTube RAG System
YouTube RAG系统会话管理
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class SessionManager:
    """Manage RAG sessions with persistence"""
    
    def __init__(self, storage_dir: str = "rag_sessions", status_callback: Optional[Callable] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.status_callback = status_callback or (lambda msg: None)
    
    def save_session(
        self,
        document: Document,
        summary: str,
        model_config: Dict[str, Any],
        persist_name: Optional[str] = None,
        chat_history: Optional[List[List[str]]] = None,
        language: str = "zh",
        documents: Optional[List[Document]] = None,
        summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Save RAG session to disk
        保存RAG会话到磁盘
        """
        try:
            if persist_name is None:
                persist_name = str(uuid.uuid4())
                
            session_path = self.storage_dir / persist_name
            if session_path.exists():
                shutil.rmtree(session_path)
            session_path.mkdir(parents=True, exist_ok=True)
            
            # Save session metadata
            serializable_history = [list(turn) for turn in (chat_history or [])]
            language = (language or "zh").lower()
            if language not in {"zh", "en"}:
                language = "en"

            documents_list = documents or [document]
            video_urls = [doc.metadata.get("source", "") for doc in documents_list]
            document_records = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents_list
            ]

            metadata = {
                "session_id": persist_name,
                "persist_name": persist_name,
                "created_at": datetime.now().isoformat(),
                "model_name": model_config.get("model_name", "gpt-5-mini"),
                "chunk_size": model_config.get("chunk_size", 1000),
                "chunk_overlap": model_config.get("chunk_overlap", 20),
                "video_url": video_urls[0] if video_urls else "",
                "video_urls": video_urls,
                "content_type": document.metadata.get("type", ""),
                "summary": summary,
                "summaries": summaries or [],
                "document_content": document.page_content,
                "documents": document_records,
                "chat_history": serializable_history,
                "language": language
            }
            
            metadata_path = session_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Build and save vector database
            chroma_path = session_path / "chroma_db"
            self._build_vector_db(documents_list, chroma_path, model_config)
            
            self.status_callback(f"✅ Session saved as '{persist_name}' / 会话已保存为 '{persist_name}'")
            return persist_name
            
        except Exception as e:
            self.status_callback(f"❌ Failed to save session / 保存会话失败: {e}")
            return None
    
    def load_session(self, persist_name: str) -> Optional[Dict[str, Any]]:
        """
        Load RAG session from disk
        从磁盘加载RAG会话
        """
        try:
            session_path = self.storage_dir / persist_name
            if not session_path.exists():
                raise FileNotFoundError(f"Session '{persist_name}' not found")
            
            # Load metadata
            metadata_path = session_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            documents_meta = metadata.get("documents")
            documents = []
            if documents_meta:
                for item in documents_meta:
                    documents.append(
                        Document(
                            page_content=item.get("content", ""),
                            metadata=item.get("metadata", {}),
                        )
                    )
            else:
                # Backward compatibility for legacy sessions
                documents.append(
                    Document(
                        page_content=metadata.get("document_content", ""),
                        metadata={
                            "source": metadata.get("video_url", ""),
                            "type": metadata.get("content_type", "")
                        }
                    )
                )
            
            # Load vector database
            chroma_path = session_path / "chroma_db"
            retriever = self._load_vector_db(chroma_path)
            
            self.status_callback(f"✅ Session '{persist_name}' loaded successfully / 会话 '{persist_name}' 加载成功")
            
            return {
                "document": documents[0],
                "summary": metadata["summary"],
                "retriever": retriever,
                "metadata": metadata,
                "chat_history": metadata.get("chat_history", []),
                "documents": documents,
                "summaries": metadata.get("summaries", []),
                "video_urls": metadata.get("video_urls", [metadata.get("video_url", "")])
            }
            
        except Exception as e:
            self.status_callback(f"❌ Failed to load session / 加载会话失败: {e}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions
        列出所有保存的会话
        """
        try:
            sessions = []
            for session_dir in self.storage_dir.iterdir():
                if session_dir.is_dir():
                    metadata_path = session_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            sessions.append({
                                "name": session_dir.name,
                                "created_at": metadata.get("created_at", ""),
                                "video_url": metadata.get("video_url", ""),
                                "model_name": metadata.get("model_name", ""),
                                "content_type": metadata.get("content_type", "")
                            })
            
            # Sort by creation time
            sessions.sort(key=lambda x: x["created_at"], reverse=True)
            return sessions
            
        except Exception as e:
            self.status_callback(f"❌ Failed to list sessions / 列出会话失败: {e}")
            return []
    
    def delete_session(self, persist_name: str) -> bool:
        """
        Delete a saved session
        删除保存的会话
        """
        try:
            session_path = self.storage_dir / persist_name
            if session_path.exists():
                shutil.rmtree(session_path)
                self.status_callback(f"✅ Session '{persist_name}' deleted / 会话 '{persist_name}' 已删除")
                return True
            else:
                self.status_callback(f"❌ Session '{persist_name}' not found / 会话 '{persist_name}' 未找到")
                return False
                
        except Exception as e:
            self.status_callback(f"❌ Failed to delete session / 删除会话失败: {e}")
            return False

    def update_chat_history(self, persist_name: str, chat_history: List[List[str]]) -> bool:
        """Update chat history for an existing session"""
        try:
            session_path = self.storage_dir / persist_name
            metadata_path = session_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"metadata for '{persist_name}' not found")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            serializable_history = [list(turn) for turn in chat_history]
            metadata['chat_history'] = serializable_history

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.status_callback(f"💾 Chat history updated for '{persist_name}'")
            return True

        except Exception as e:
            self.status_callback(f"❌ Failed to update chat history for '{persist_name}': {e}")
            return False

    def append_to_session(
        self,
        persist_name: str,
        documents: List[Document],
        new_documents: List[Document],
        summaries: List[Dict[str, Any]],
        combined_summary: str,
        combined_document: Document,
        chat_history: List[List[str]],
        language: str,
        model_config: Dict[str, Any]
    ) -> bool:
        """Append documents to an existing session without rebuilding the entire vector store."""
        try:
            session_path = self.storage_dir / persist_name
            metadata_path = session_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Session metadata for '{persist_name}' not found")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            serializable_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]

            metadata['documents'] = serializable_docs
            metadata['document_content'] = combined_document.page_content
            metadata['content_type'] = combined_document.metadata.get('type', metadata.get('content_type', ''))
            metadata['summary'] = combined_summary
            metadata['summaries'] = summaries
            metadata['chat_history'] = [list(turn) for turn in chat_history]
            metadata['language'] = language
            metadata['video_urls'] = [doc.metadata.get('source', '') for doc in documents]
            if metadata['video_urls']:
                metadata['video_url'] = metadata['video_urls'][0]

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            chroma_path = session_path / "chroma_db"
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=model_config.get("chunk_size", 1000),
                chunk_overlap=model_config.get("chunk_overlap", 20)
            )

            new_chunks = text_splitter.split_documents(new_documents)
            if new_chunks:
                embeddings = OpenAIEmbeddings()
                vector_store = Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=embeddings
                )
                vector_store.add_documents(new_chunks)
                vector_store.persist()

            self.status_callback(f"✅ Session '{persist_name}' updated with new content")
            return True

        except Exception as e:
            self.status_callback(f"❌ Failed to append to session '{persist_name}': {e}")
            return False
    
    def _build_vector_db(self, documents: List[Document], persist_path: Path, model_config: Dict[str, Any]):
        """Build and persist vector database"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=model_config.get("chunk_size", 1000),
            chunk_overlap=model_config.get("chunk_overlap", 20)
        )
        chunks = text_splitter.split_documents(documents)
        
        # Vectorization and storage
        embeddings = OpenAIEmbeddings()
        Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=str(persist_path)
        )
    
    def _build_vector_db_and_get_retriever(self, documents: List[Document], model_config: Dict[str, Any]):
        """Build vector database in memory and return retriever"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=model_config.get("chunk_size", 1000),
            chunk_overlap=model_config.get("chunk_overlap", 20)
        )
        chunks = text_splitter.split_documents(documents)
        
        # Vectorization and storage
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        return vector_store.as_retriever()
    
    def _load_vector_db(self, persist_path: Path):
        """Load vector database and return retriever"""
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )
        return vector_store.as_retriever()
