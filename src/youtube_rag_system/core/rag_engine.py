"""
Main RAG engine for YouTube video Q&A
YouTube视频问答主引擎
"""

import os
from typing import Optional, Callable, Dict, Any

from langchain.schema import Document, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .content_processor import ContentProcessor
from .session_manager import SessionManager


class YouTubeRAG:
    """YouTube Video RAG Q&A System"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 20, 
                 model_name: str = "gpt-5-mini", language: str = "zh",
                 status_callback: Optional[Callable] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.status_callback = status_callback or (lambda msg: None)
        self.language = (language or "zh").lower()
        if self.language not in {"zh", "en"}:
            self.language = "en"
        
        self._check_openai_key()
        
        # Initialize components
        self.content_processor = ContentProcessor(model_name, status_callback)
        self.session_manager = SessionManager(status_callback=status_callback)
    
    def _check_openai_key(self):
        """Check OpenAI API key"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    def process_video(self, url: str, allow_transcription: bool = True, 
                     save_summary: bool = False, save_original: bool = False) -> Dict[str, Any]:
        """
        Process YouTube video and return RAG components
        处理YouTube视频并返回RAG组件
        """
        # Get video content
        document = self.content_processor.get_video_content(url, allow_transcription)
        
        # Generate summary
        summary = self.content_processor.generate_summary(document, language=self.language)

        # Save files if requested
        saved_files = []
        if save_summary and summary:
            summary_file = self.content_processor.save_summary(summary, document, save_to_file=True)
            if summary_file:
                saved_files.append(f"Summary: {summary_file}")
                
        if save_original:
            original_file = self.content_processor.save_original_text(document, save_to_file=True)
            if original_file:
                saved_files.append(f"Original: {original_file}")
        
        # Build knowledge base and create QA chain
        model_config = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

        retriever = self.session_manager._build_vector_db_and_get_retriever([document], model_config)
        qa_chain = self.create_qa_chain(retriever)

        summaries = [{"video_url": url, "summary": summary}]

        # Auto-save session
        from ..utils.file_utils import extract_video_id
        video_id = extract_video_id(url)
        session_name = self.session_manager.save_session(
            document,
            summary,
            model_config,
            video_id,
            chat_history=[],
            language=self.language,
            documents=[document],
            summaries=summaries
        )

        loaded_session = self.load_session(session_name)
        if not loaded_session:
            raise ValueError("Failed to load session after saving")

        qa_chain = self.create_qa_chain(loaded_session["retriever"])
        loaded_session["qa_chain"] = qa_chain
        loaded_session["saved_files"] = saved_files
        loaded_session["session_name"] = session_name
        loaded_session["chat_history"] = []
        loaded_session["language"] = self.language
        loaded_session["summary"] = summary
        loaded_session["summaries"] = summaries
        loaded_session["documents"] = [document]
        loaded_session["document"] = document
        loaded_session["video_urls"] = [url]
        return loaded_session
    
    def create_qa_chain(self, retriever):
        """Create Q&A chain"""
        system_prompt = """Answer user questions based on the following context.
If you don't know the answer, say "I don't know" and don't make up answers.
Please answer in the same language as the question.

基于以下上下文回答用户问题。
如果不知道答案，请说"我不知道"，不要编造答案。
请用与问题相同的语言回答。

Context / 上下文: {context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=1,
            model_kwargs={"max_completion_tokens": 2048}
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        return create_retrieval_chain(retriever, question_answer_chain)

    def _format_history(self, history: Optional[list]) -> list:
        """Convert UI chat history into LangChain message list"""
        if not history:
            return []
        messages = []
        for user_turn, assistant_turn in history:
            if user_turn and user_turn not in {"System", "system"}:
                messages.append(HumanMessage(content=str(user_turn)))
            if assistant_turn:
                messages.append(AIMessage(content=str(assistant_turn)))
        return messages

    def generate_analysis_report(
        self,
        session_data: Dict[str, Any],
        language: Optional[str] = None
    ) -> str:
        """Generate a detailed analysis report for a loaded session"""
        if not session_data:
            raise ValueError("Session data is required")

        document = session_data.get("document")
        if document is None:
            raise ValueError("Session data must include a document")

        summary = session_data.get("summary")
        metadata = session_data.get("metadata", {})
        language_choice = (language or metadata.get("language") or self.language or "en").lower()
        if language_choice not in {"zh", "en"}:
            language_choice = "en"

        return self.content_processor.generate_analysis_report(
            document=document,
            summary=summary,
            metadata=metadata,
            language=language_choice
        )

    def ask_question(self, qa_chain, question: str, history: Optional[list] = None) -> str:
        """Ask a question using the QA chain"""
        try:
            formatted_history = self._format_history(history)
            result = qa_chain.invoke({"input": question, "chat_history": formatted_history})
            return result['answer']
        except Exception as e:
            raise Exception(f"Error occurred: {e}")
    
    # Session management methods
    def save_session(self, document: Document, summary: str, persist_name: Optional[str] = None,
                     chat_history: Optional[list] = None) -> Optional[str]:
        """Save current session"""
        model_config = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        return self.session_manager.save_session(
            document,
            summary,
            model_config,
            persist_name,
            chat_history=chat_history,
            language=self.language
        )

    def load_session(self, persist_name: str) -> Optional[Dict[str, Any]]:
        """Load saved session"""
        session_data = self.session_manager.load_session(persist_name)
        if session_data:
            # Update model settings from loaded session
            metadata = session_data["metadata"]
            self.model_name = metadata["model_name"]
            self.chunk_size = metadata["chunk_size"]
            self.chunk_overlap = metadata["chunk_overlap"]
            self.language = metadata.get("language", self.language)
            
            # Create QA chain with loaded retriever
            qa_chain = self.create_qa_chain(session_data["retriever"])
            session_data["qa_chain"] = qa_chain
            session_data["session_name"] = persist_name
            session_data["language"] = self.language
        
        return session_data
    
    def list_sessions(self) -> list:
        """List all saved sessions"""
        return self.session_manager.list_sessions()
    
    def delete_session(self, persist_name: str) -> bool:
        """Delete a saved session"""
        return self.session_manager.delete_session(persist_name)

    def update_chat_history(self, persist_name: str, chat_history: list) -> bool:
        """Persist chat history updates"""
        return self.session_manager.update_chat_history(persist_name, chat_history)

    def add_video_to_session(self, session_data: Dict[str, Any], url: str, allow_transcription: bool = True) -> Dict[str, Any]:
        """Append a new video's content to an existing session"""
        if not session_data:
            raise ValueError("No active session to extend")

        document = self.content_processor.get_video_content(url, allow_transcription)
        summary = self.content_processor.generate_summary(document, language=self.language)

        existing_documents = session_data.get("documents") or [session_data.get("document")]
        documents = [doc for doc in existing_documents if doc is not None]
        documents.append(document)

        summaries = session_data.get("summaries") or []
        summaries = list(summaries)
        summaries.append({"video_url": url, "summary": summary})

        combined_summary = session_data.get("summary", "")
        if combined_summary:
            combined_summary += "\n\n" + ("-" * 50) + "\n\n"
        combined_summary += summary

        combined_content = "\n\n".join(doc.page_content for doc in documents)
        primary_source = session_data.get("video_urls", [url])[0] if session_data.get("video_urls") else url
        combined_document = Document(
            page_content=combined_content,
            metadata={
                "source": primary_source,
                "type": "combined"
            }
        )

        model_config = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

        persist_name = session_data.get("session_name") or session_data.get("metadata", {}).get("persist_name")
        if not persist_name:
            raise ValueError("Session persist name missing")

        chat_history = session_data.get("chat_history", [])

        append_ok = self.session_manager.append_to_session(
            persist_name=persist_name,
            documents=documents,
            new_documents=[document],
            summaries=summaries,
            combined_summary=combined_summary,
            combined_document=combined_document,
            chat_history=chat_history,
            language=self.language,
            model_config=model_config
        )

        if not append_ok:
            raise ValueError("Failed to update session with new video")

        refreshed = self.load_session(persist_name)
        if not refreshed:
            raise ValueError("Failed to reload session after adding video")

        qa_chain = self.create_qa_chain(refreshed["retriever"])
        refreshed["qa_chain"] = qa_chain
        refreshed["session_name"] = persist_name
        refreshed["chat_history"] = chat_history
        refreshed["summary"] = combined_summary
        refreshed["summaries"] = summaries
        refreshed["documents"] = documents
        refreshed["document"] = combined_document
        refreshed["video_urls"] = refreshed.get("video_urls", [])
        if url not in refreshed["video_urls"]:
            refreshed["video_urls"].append(url)

        return {
            "session": refreshed,
            "new_summary": summary
        }
