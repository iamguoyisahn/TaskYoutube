"""
Gradio web interface for YouTube RAG System
YouTube RAG系统的Gradio网页界面
"""

import os
import gradio as gr
from typing import Optional

from langchain.schema import Document

from ..core.rag_engine import YouTubeRAG
from ..utils.validators import validate_api_key, validate_youtube_url


class YouTubeRAGInterface:
    """Simplified Gradio interface for YouTube RAG System"""
    
    def __init__(self):
        self.state = "api_key"
        self.rag_system: Optional[YouTubeRAG] = None
        self.qa_chain = None
        self.current_session = None
        self.language = "zh"
    
    def reset_system(self):
        """Reset the system state"""
        self.state = "api_key"
        self.rag_system = None
        self.qa_chain = None
        self.current_session = None
        self.language = "zh"
    
    def chat_response(self, message: str, history: list) -> list:
        """Main chat response handler"""
        message = message.strip()
        
        # Handle special commands
        if message.lower() in ['reset', '重置']:
            self.reset_system()
            return self._get_welcome_message(history)
        
        if message.lower() in ['exit', 'quit', '退出']:
            return history + [(message, "👋 Goodbye! / 再见！")]
        
        # Route to appropriate handler based on state
        if self.state == "start":
            return self._handle_start(message, history)
        elif self.state == "api_key":
            return self._handle_api_key(message, history)
        elif self.state == "language_choice":
            return self._handle_language_choice(message, history)
        elif self.state == "action_choice":
            return self._handle_action_choice(message, history)
        elif self.state == "new_video":
            return self._handle_new_video(message, history)
        elif self.state == "session_select":
            return self._handle_session_select(message, history)
        elif self.state == "ready":
            return self._handle_questions(message, history)
        else:
            return history + [(message, "❌ 系统错误，请输入 'reset' 重新开始 / System error, type 'reset' to restart")]
    
    def _get_welcome_message(self, history: list) -> list:
        """Get welcome message"""
        welcome = """👋 欢迎使用YouTube RAG问答系统！/ Welcome to YouTube RAG Q&A System!

🔐 请输入您的OpenAI API密钥 (格式: sk-...)
Please enter your OpenAI API key (format: sk-...)"""
        
        if history and isinstance(history[-1], (list, tuple)) and history[-1] == ("System", welcome):
            self.state = "api_key"
            return history

        self.state = "api_key"
        return history + [("System", welcome)]
    
    def _handle_start(self, message: str, history: list) -> list:
        """Handle initial state"""
        return self._get_welcome_message(history)
    
    def _handle_api_key(self, message: str, history: list) -> list:
        """Handle API key input"""
        if not validate_api_key(message):
            error_msg = """❌ API密钥格式不正确！/ Invalid API key format!
请重新输入 (格式: sk-...) / Please re-enter (format: sk-...)"""
            return history + [(message, error_msg)]
        
        # Set API key
        os.environ["OPENAI_API_KEY"] = message

        language_msg = """✅ API密钥设置成功！/ API key set successfully!

🌐 请选择摘要语言 / Please choose summary language:
1. 简体中文 / Simplified Chinese
2. English / 英文

请输入编号或语言名称 / Enter number or language name:"""

        self.state = "language_choice"
        return history + [(message, language_msg)]

    def _language_display(self) -> str:
        return "简体中文" if self.language == "zh" else "English"

    def _action_choice_prompt(self) -> str:
        return """请选择操作 / Please choose action:
1. 🆕 分析新视频 / Analyze new video
2. 📂 加载已保存的会话 / Load saved session

请输入 1 或 2 / Please enter 1 or 2:"""

    def _handle_language_choice(self, message: str, history: list) -> list:
        """Handle language selection"""
        choice = message.strip().lower()

        if choice in {"1", "zh", "cn", "中文", "chinese", "简体", "简体中文"}:
            self.language = "zh"
        elif choice in {"2", "en", "english", "英文"}:
            self.language = "en"
        else:
            error_msg = """❌ 无法识别的语言选项 / Unrecognized language option
请输入 1 (中文) 或 2 (English) / Please enter 1 (Chinese) or 2 (English)"""
            return history + [(message, error_msg)]

        confirm_msg = (
            f"✅ 语言已设置为 {self._language_display()} / Language set to {self._language_display()}\n\n"
            f"{self._action_choice_prompt()}"
        )

        self.state = "action_choice"
        return history + [(message, confirm_msg)]
    
    def _handle_action_choice(self, message: str, history: list) -> list:
        """Handle action choice"""
        choice = message.strip()
        
        if choice == "1":
            video_msg = """🎥 请输入YouTube视频链接 / Please enter YouTube video URL:

支持格式 / Supported formats:
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID"""
            
            self.state = "new_video"
            return history + [(message, video_msg)]
        
        elif choice == "2":
            try:
                # Initialize RAG system to list sessions
                temp_rag = YouTubeRAG()
                sessions = temp_rag.list_sessions()
                
                if not sessions:
                    no_sessions_msg = """📋 没有找到已保存的会话 / No saved sessions found

请选择分析新视频 / Please choose to analyze new video:
🎥 请输入YouTube视频链接 / Please enter YouTube video URL:"""
                    
                    self.state = "new_video"
                    return history + [(message, no_sessions_msg)]
                
                # Show available sessions
                sessions_text = "📋 可用会话 / Available Sessions:\n\n"
                for i, session in enumerate(sessions[:5], 1):  # Show max 5 sessions
                    created_time = session['created_at'][:19].replace('T', ' ')
                    sessions_text += f"{i}. {session['name']}\n"
                    sessions_text += f"   🕒 {created_time}\n"
                    sessions_text += f"   🎥 {session['video_url'][:60]}...\n\n"
                
                sessions_msg = f"""{sessions_text}
请输入会话编号 (1-{min(len(sessions), 5)}) 或会话名称:
Please enter session number (1-{min(len(sessions), 5)}) or session name:"""
                
                self.sessions_list = sessions
                self.state = "session_select"
                return history + [(message, sessions_msg)]
                
            except Exception as e:
                error_msg = f"""❌ 无法列出会话 / Cannot list sessions: {e}

请选择分析新视频 / Please choose to analyze new video:
🎥 请输入YouTube视频链接 / Please enter YouTube video URL:"""
                
                self.state = "new_video"
                return history + [(message, error_msg)]
        
        else:
            error_msg = """❌ 请输入 1 或 2 / Please enter 1 or 2
1. 分析新视频 / Analyze new video
2. 加载已保存的会话 / Load saved session"""
            
            return history + [(message, error_msg)]
    
    def _handle_new_video(self, message: str, history: list) -> list:
        """Handle new video processing"""
        url = message.strip()
        
        if not validate_youtube_url(url):
            error_msg = """❌ YouTube链接格式不正确！/ Invalid YouTube URL format!
请输入有效的YouTube链接 / Please enter a valid YouTube URL:"""
            return history + [(message, error_msg)]
        
        try:
            # Show processing message
            processing_msg = """🔄 正在处理视频... / Processing video...

这可能需要几分钟时间，请稍等 / This may take a few minutes, please wait"""
            
            history = history + [(message, processing_msg)]
            
            # Initialize RAG system and process video
            self.rag_system = YouTubeRAG(language=self.language, status_callback=lambda msg: None)
            self.rag_system.language = self.language
            result = self.rag_system.process_video(url, allow_transcription=True)
            
            self.qa_chain = result["qa_chain"]
            self.current_session = result
            self.current_session.setdefault('chat_history', [])
            self.current_session['language'] = self.language
            
            storage_path = f"rag_sessions/{result['session_name']}"

            language_label = self._language_display()
            success_msg = f"""✅ 视频处理完成！/ Video processing completed!

📄 **视频摘要 / Video Summary:**
{'-' * 50}
{result['summary']}
{'-' * 50}

💾 **会话已保存为:** {result['session_name']}
**Session saved as:** {result['session_name']}
📂 **存储路径 / Storage Path:** {storage_path}

🈯 **摘要语言 / Summary Language:** {language_label}

🤖 现在您可以提问了！/ Now you can ask questions!

💡 **特殊命令 / Special Commands:**
- 'sessions' - 查看所有会话 / View all sessions
- 'save as [名称]' - 另存会话 / Save session as
- 'save summary' - 保存摘要到TXT / Save summary to TXT
- 'save subtitles' - 保存字幕/原文到TXT / Save subtitles/original text to TXT
- 'add video [链接]' - 继续追加视频 / Append another video
- 'reset' - 重新开始 / Restart"""
            
            self.state = "ready"
            return history + [("System", success_msg)]
            
        except Exception as e:
            error_msg = f"""❌ 视频处理失败！/ Video processing failed: {e}

请重新输入YouTube链接 / Please re-enter YouTube URL:"""
            
            return history + [("System", error_msg)]
    
    def _handle_session_select(self, message: str, history: list) -> list:
        """Handle session selection"""
        choice = message.strip()
        
        try:
            # Try to parse as session number
            if choice.isdigit():
                session_idx = int(choice) - 1
                if 0 <= session_idx < len(self.sessions_list):
                    session_name = self.sessions_list[session_idx]['name']
                else:
                    error_msg = f"""❌ 无效编号！/ Invalid number!
请输入 1-{len(self.sessions_list)} / Please enter 1-{len(self.sessions_list)}"""
                    return history + [(message, error_msg)]
            else:
                # Try to find by name
                session_name = choice
                if not any(s['name'] == session_name for s in self.sessions_list):
                    error_msg = "❌ 会话名称未找到！/ Session name not found!"
                    return history + [(message, error_msg)]
            
            # Load session
            self.rag_system = YouTubeRAG()
            session_data = self.rag_system.load_session(session_name)
            
            if not session_data:
                error_msg = f"""❌ 加载会话失败！/ Failed to load session!
请重新选择 / Please choose again:"""
                return history + [(message, error_msg)]
            
            self.qa_chain = session_data["qa_chain"]
            self.current_session = session_data
            self.current_session.setdefault('chat_history', [])
            self.language = session_data.get('language') or session_data.get('metadata', {}).get('language', 'zh')
            self.rag_system.language = self.language
            self.current_session['language'] = self.language
            
            storage_path = f"rag_sessions/{session_name}"
            language_label = self._language_display()

            success_msg = f"""✅ 会话加载成功！/ Session loaded successfully!

📋 **会话:** {session_name}
📄 **摘要 / Summary:**
{'-' * 50}
{session_data['summary']}
{'-' * 50}

🈯 **摘要语言 / Summary Language:** {language_label}
📂 **存储路径 / Storage Path:** {storage_path}

🤖 您可以继续提问！/ You can continue asking questions!

💡 **特殊命令 / Special Commands:**
- 'add video [链接]' - 继续追加视频 / Append another video
- 'save summary' / 'save subtitles' - 导出资料 / Export data
- 'sessions' / 'save as [名称]' - 管理会话 / Manage sessions
- 'reset' - 重新开始 / Restart"""
            
            self.state = "ready"

            enriched_history = list(history)
            for turn in self.current_session.get('chat_history', []):
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    enriched_history.append((turn[0], turn[1]))
            enriched_history.append((message, success_msg))

            return enriched_history
            
        except Exception as e:
            error_msg = f"""❌ 加载会话出错！/ Error loading session: {e}
请重新选择 / Please choose again:"""
            return history + [(message, error_msg)]
    
    def _handle_questions(self, message: str, history: list) -> list:
        """Handle Q&A questions and session management"""
        message_lower = message.lower()
        
        # Handle session management commands
        if message_lower == 'sessions':
            try:
                sessions = self.rag_system.list_sessions()
                if not sessions:
                    sessions_msg = "📋 没有已保存的会话 / No saved sessions"
                else:
                    sessions_text = "📋 **已保存的会话 / Saved Sessions:**\n\n"
                    for session in sessions[:10]:  # Show max 10
                        created_time = session['created_at'][:19].replace('T', ' ')
                        sessions_text += f"• **{session['name']}**\n"
                        sessions_text += f"  🕒 {created_time}\n"
                        sessions_text += f"  🎥 {session['video_url'][:50]}...\n\n"
                    sessions_msg = sessions_text
                
                return history + [(message, sessions_msg)]
                
            except Exception as e:
                error_msg = f"❌ 列出会话失败 / Failed to list sessions: {e}"
                return history + [(message, error_msg)]
        
        elif message_lower.startswith('save as '):
            session_name = message[8:].strip()
            if not session_name:
                error_msg = """❌ 请提供会话名称 / Please provide session name
格式 / Format: save as [会话名称]"""
                return history + [(message, error_msg)]
            
            try:
                if self.current_session and 'document' in self.current_session:
                    saved_name = self.rag_system.save_session(
                        self.current_session['document'],
                        self.current_session['summary'],
                        session_name,
                        chat_history=self.current_session.get('chat_history')
                    )
                    if saved_name:
                        storage_path = f"rag_sessions/{saved_name}"
                        language_label = self._language_display()
                        success_msg = (
                            f"✅ 会话已保存为 '{saved_name}' / Session saved as '{saved_name}'\n"
                            f"📂 存储路径 / Storage Path: {storage_path}\n"
                            f"🈯 摘要语言 / Summary Language: {language_label}"
                        )
                    else:
                        success_msg = "❌ 保存失败 / Save failed"
                else:
                    success_msg = "❌ 没有可保存的会话 / No session to save"
                
                return history + [(message, success_msg)]
                
            except Exception as e:
                error_msg = f"❌ 保存失败 / Save failed: {e}"
                return history + [(message, error_msg)]
        
        elif message_lower in {'save summary', '保存摘要', '保存summary'}:
            if not self.current_session:
                error_msg = "❌ 当前没有会话可保存摘要 / No active session to save summary"
                return history + [(message, error_msg)]

            summary = self.current_session.get('summary')
            if not summary:
                summaries = self.current_session.get('summaries') or []
                summary = "\n\n".join(
                    item.get('summary', '') for item in summaries if item.get('summary')
                )

            document = self.current_session.get('document')
            if document is None:
                docs = self.current_session.get('documents') or []
                if docs:
                    document = docs[0]
            if document is None:
                metadata = self.current_session.get('metadata') or {}
                content = metadata.get('document_content')
                if content:
                    document = Document(
                        page_content=content,
                        metadata={
                            'source': metadata.get('video_url', ''),
                            'type': metadata.get('content_type', 'content')
                        }
                    )

            if not summary or not document:
                error_msg = "❌ 未找到摘要或原始文档 / Missing summary or document"
                return history + [(message, error_msg)]

            filepath = self.rag_system.content_processor.save_summary(summary, document, save_to_file=True)
            if filepath:
                response = f"✅ 摘要已保存至 {filepath} / Summary saved to {filepath}"
            else:
                response = "❌ 摘要保存失败 / Failed to save summary"
            return history + [(message, response)]

        elif message_lower in {'save subtitles', 'save original', '保存字幕', '保存原文'}:
            if not self.current_session:
                error_msg = "❌ 当前没有会话可保存字幕 / No active session to save subtitles"
                return history + [(message, error_msg)]

            document = self.current_session.get('document')
            if document is None:
                docs = self.current_session.get('documents') or []
                if docs:
                    document = docs[0]
            if document is None:
                metadata = self.current_session.get('metadata') or {}
                content = metadata.get('document_content')
                if content:
                    document = Document(
                        page_content=content,
                        metadata={
                            'source': metadata.get('video_url', ''),
                            'type': metadata.get('content_type', 'content')
                        }
                    )

            if not document:
                error_msg = "❌ 未找到原始文档 / Missing document"
                return history + [(message, error_msg)]

            filepath = self.rag_system.content_processor.save_original_text(document, save_to_file=True)
            if filepath:
                response = f"✅ 字幕/原文已保存至 {filepath} / Subtitles saved to {filepath}"
            else:
                response = "❌ 字幕保存失败 / Failed to save subtitles"
            return history + [(message, response)]

        elif message_lower.startswith('add video') or message_lower.startswith('添加视频'):
            if not self.current_session:
                error_msg = "❌ 当前没有活动会话，无法追加视频 / No active session to extend"
                return history + [(message, error_msg)]

            url = message
            if message_lower.startswith('add video'):
                url = message[len('add video'):].strip()
            elif message_lower.startswith('添加视频'):
                url = message[len('添加视频'):].strip()

            if not url:
                error_msg = """❌ 请在命令后提供YouTube链接 / Please provide a YouTube URL
格式 / Format: add video https://youtu.be/..."""
                return history + [(message, error_msg)]

            if not validate_youtube_url(url):
                error_msg = "❌ YouTube链接格式不正确 / Invalid YouTube URL"
                return history + [(message, error_msg)]

            try:
                result = self.rag_system.add_video_to_session(self.current_session, url)
                new_session = result['session']
                self.current_session = new_session
                self.qa_chain = new_session['qa_chain']

                new_summary = result.get('new_summary', '')
                preview = new_summary.strip()
                if len(preview) > 400:
                    preview = preview[:400] + '...'

                success_msg = f"""✅ 已追加新视频并更新知识库 / Video added to knowledge base

📺 **链接 / URL:** {url}
🧠 **当前视频总数 / Videos in session:** {len(self.current_session.get('documents', []))}

📝 **新增摘要预览 / New Summary Preview:**
{preview if preview else '（暂无摘要文本 / No summary text）'}"""

                return history + [(message, success_msg)]

            except Exception as e:
                error_msg = f"❌ 追加视频失败 / Failed to add video: {e}"
                return history + [(message, error_msg)]

        else:
            # Regular Q&A question
            try:
                answer = self.rag_system.ask_question(self.qa_chain, message, history)
                if self.current_session is not None:
                    chat_history = self.current_session.setdefault('chat_history', [])
                    chat_history.append((message, answer))
                    session_name = self.current_session.get('session_name')
                    if session_name:
                        # Persist chat history asynchronously (failure won't stop reply)
                        try:
                            self.rag_system.update_chat_history(session_name, chat_history)
                        except Exception:
                            pass
                return history + [(message, answer)]
                
            except Exception as e:
                error_msg = f"❌ 回答生成失败 / Answer generation failed: {e}"
                return history + [(message, error_msg)]
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="YouTube RAG Chat", theme="Taithrah/Minimal") as interface:
            gr.Markdown("""
            # 🎥 YouTube RAG 问答系统 / YouTube RAG Q&A System
            
            轻量级YouTube视频分析和智能问答工具  
            Lightweight YouTube video analysis and intelligent Q&A tool
            """)
            
            chatbot = gr.Chatbot(
                value=[("System", """👋 欢迎使用YouTube RAG问答系统！/ Welcome to YouTube RAG Q&A System!

🔐 请输入您的OpenAI API密钥 (格式: sk-...)
Please enter your OpenAI API key (format: sk-...)""")],
                height=600,
                show_label=False
            )
            
            msg = gr.Textbox(
                placeholder="请输入消息... / Enter your message...",
                show_label=False
            )
            
            def respond(message: str, chat_history: list):
                if not message.strip():
                    return chat_history, ""
                
                new_history = self.chat_response(message, chat_history)
                return new_history, ""
            
            def reset_chat():
                self.reset_system()
                return [("System", """👋 欢迎使用YouTube RAG问答系统！/ Welcome to YouTube RAG Q&A System!

🔐 请输入您的OpenAI API密钥 (格式: sk-...)
Please enter your OpenAI API key (format: sk-...)""")], ""
            
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            
            with gr.Row():
                reset_btn = gr.Button("🔄 重置 / Reset", variant="secondary")
                reset_btn.click(reset_chat, outputs=[chatbot, msg])
            
            gr.Markdown("""
            ### 💡 使用说明 / Instructions:
            1. 输入OpenAI API密钥 / Enter OpenAI API key
            2. 选择分析新视频或加载已保存会话 / Choose to analyze new video or load saved session
            3. 开始提问！/ Start asking questions!
            4. 使用 `add video <URL>` 可继续扩展知识库 / Use `add video <URL>` to expand the knowledge base
            
            ### 🔧 功能特性 / Features:
            - 🎯 智能字幕提取 / Intelligent subtitle extraction
            - 📝 自动视频摘要 / Automatic video summarization  
            - 🤖 基于RAG的问答 / RAG-based Q&A
            - 💾 会话持久化存储 / Session persistence
            - 🌐 支持多语言 / Multi-language support
            - 📄 一键导出摘要与字幕 / One-click summary & subtitle export
            """)
        
        return interface


def create_interface() -> gr.Blocks:
    """Create and return Gradio interface"""
    rag_interface = YouTubeRAGInterface()
    return rag_interface.create_interface()
