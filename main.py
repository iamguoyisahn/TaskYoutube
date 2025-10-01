#!/usr/bin/env python3.12
"""
YouTube RAG System - Main Entry Point
YouTube RAG系统 - 主入口程序

A lightweight YouTube video Q&A tool with persistent session management.
轻量级YouTube视频问答工具，支持持久化会话管理。
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from youtube_rag_system import YouTubeRAG
from youtube_rag_system.ui.gradio_interface import create_interface


def main():
    """Main entry point"""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="YouTube RAG System - YouTube视频RAG问答工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  python3.12 main.py --ui                                    # Launch web interface / 启动网页界面
  python3.12 main.py --url "https://youtube.com/watch?v=..."  # CLI mode / 命令行模式
  python3.12 main.py --list-sessions                        # List saved sessions / 列出保存的会话
        """
    )
    
    parser.add_argument(
        "--ui", 
        action="store_true",
        help="Launch Gradio web interface / 启动Gradio网页界面"
    )
    
    parser.add_argument(
        "--url", 
        type=str,
        help="YouTube video URL for CLI mode / 命令行模式的YouTube视频URL"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-3.5-turbo",
        help="OpenAI model name / OpenAI模型名称 (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Text chunk size / 文本块大小 (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=20,
        help="Text chunk overlap / 文本块重叠 (default: 20)"
    )
    
    parser.add_argument(
        "--list-sessions", 
        action="store_true",
        help="List all saved sessions / 列出所有保存的会话"
    )
    
    parser.add_argument(
        "--load-session", 
        type=str,
        help="Load a saved session by name / 根据名称加载保存的会话"
    )
    
    parser.add_argument(
        "--delete-session", 
        type=str,
        help="Delete a saved session by name / 根据名称删除保存的会话"
    )

    parser.add_argument(
        "--export-analysis",
        type=str,
        help="Generate a detailed analysis file for a saved session / 为保存的会话生成详细分析文件"
    )

    parser.add_argument(
        "--analysis-output",
        type=str,
        help="Output path for generated analysis (default: analysis/<session>_analysis.md) / 分析文件输出路径"
    )

    parser.add_argument(
        "--analysis-language",
        type=str,
        choices=["zh", "en"],
        help="Language for analysis output (default matches session) / 分析输出语言"
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable Whisper transcription fallback when subtitles are missing / 若无字幕则不转录音频"
    )
    
    args = parser.parse_args()
    
    # Default to UI mode if no arguments provided
    if len(sys.argv) == 1:
        args.ui = True
    
    try:
        if args.ui:
            # Launch web interface
            print("🚀 Launching YouTube RAG System Web Interface...")
            print("🚀 启动YouTube RAG系统网页界面...")
            interface = create_interface()
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                show_error=True
            )
            
        elif args.list_sessions:
            # List sessions
            rag = YouTubeRAG()
            sessions = rag.list_sessions()
            if not sessions:
                print("📋 No saved sessions found / 没有找到保存的会话")
            else:
                print(f"📋 Found {len(sessions)} saved sessions / 找到{len(sessions)}个保存的会话:\n")
                for i, session in enumerate(sessions, 1):
                    created_time = session['created_at'][:19].replace('T', ' ')
                    print(f"{i}. {session['name']}")
                    print(f"   🕒 Created: {created_time}")
                    print(f"   🎥 Video: {session['video_url']}")
                    print(f"   🤖 Model: {session['model_name']}")
                    print()
                    
        elif args.delete_session:
            # Delete session
            rag = YouTubeRAG()
            if rag.delete_session(args.delete_session):
                print(f"✅ Session '{args.delete_session}' deleted successfully")
            else:
                print(f"❌ Failed to delete session '{args.delete_session}'")
                
        elif args.export_analysis:
            rag = YouTubeRAG(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                model_name=args.model,
                status_callback=print
            )

            session_name = args.export_analysis
            session_data = rag.load_session(session_name)
            if not session_data:
                print(f"❌ Failed to load session '{session_name}'")
                sys.exit(1)

            metadata = session_data.get("metadata", {})
            default_language = metadata.get("language") or rag.language
            language_choice = args.analysis_language or default_language
            analysis_text = rag.generate_analysis_report(session_data, language=language_choice)

            output_path = args.analysis_output
            if output_path:
                target_path = Path(output_path)
            else:
                target_path = Path("analysis") / f"{session_name}_analysis.md"

            target_path.parent.mkdir(parents=True, exist_ok=True)

            header_lines = [
                f"# {session_name} Analysis",
                "",
            ]

            video_urls = session_data.get("video_urls") or metadata.get("video_urls") or []
            if video_urls:
                header_lines.append(f"Source Video: {video_urls[0]}")
                header_lines.append("")

            header_lines.append(analysis_text)
            target_path.write_text("\n".join(header_lines), encoding="utf-8")

            print(f"✅ Analysis saved to {target_path}")

        elif args.load_session:
            # Load and interact with session
            rag = YouTubeRAG(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                model_name=args.model,
                status_callback=print
            )
            
            session_data = rag.load_session(args.load_session)
            if not session_data:
                print(f"❌ Failed to load session '{args.load_session}'")
                sys.exit(1)
            
            qa_chain = session_data["qa_chain"]
            print(f"\n✅ Session '{args.load_session}' loaded successfully!")
            print(f"\n📄 Summary:\n{'-' * 50}")
            print(session_data['summary'])
            print('-' * 50)
            
            # Interactive Q&A
            print("\n🤖 Q&A system ready! Type 'quit' to exit / 问答系统就绪！输入 'quit' 退出")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\n❓ Ask a question / 请提问: ").strip()
                    if question.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if not question:
                        continue
                    
                    print("🤔 Thinking... / 思考中...")
                    answer = rag.ask_question(qa_chain, question)
                    print(f"\n💡 {answer}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("\n👋 Goodbye! / 再见！")
            
        elif args.url:
            # CLI mode with URL
            def status_print(msg):
                print(msg)
            
            rag = YouTubeRAG(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                model_name=args.model,
                status_callback=status_print
            )
            
            print(f"🎥 Processing video: {args.url}")
            allow_transcription = not args.no_transcription
            result = rag.process_video(
                args.url,
                allow_transcription=allow_transcription,
                save_summary=True,
                save_original=True
            )
            
            print(f"\n📄 Video Summary:\n{'-' * 50}")
            print(result['summary'])
            print('-' * 50)
            
            if result['session_name']:
                print(f"\n💾 Session saved as: {result['session_name']}")
            
            # Interactive Q&A
            qa_chain = result['qa_chain']
            print("\n🤖 Q&A system ready! Type 'quit' to exit / 问答系统就绪！输入 'quit' 退出")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\n❓ Ask a question / 请提问: ").strip()
                    if question.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if not question:
                        continue
                    
                    print("🤔 Thinking... / 思考中...")
                    answer = rag.ask_question(qa_chain, question)
                    print(f"\n💡 {answer}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("\n👋 Goodbye! / 再见！")
            
        else:
            # No valid arguments provided
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
