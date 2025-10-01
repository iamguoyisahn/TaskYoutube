"""
Content processing module for YouTube videos
YouTube视频内容处理模块
"""

import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from openai import OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..utils.file_utils import extract_video_id, save_text_file


class ContentProcessor:
    """Process YouTube video content (subtitles/transcription and summarization)"""
    
    def __init__(self, model_name: str = "gpt-5-mini", status_callback: Optional[Callable] = None):
        self.model_name = model_name
        self.status_callback = status_callback or (lambda msg: None)
    
    def get_video_content(self, url: str, allow_transcription: bool = True) -> Document:
        """
        Get video content, prioritize subtitles, fallback to audio transcription
        获取视频内容，优先字幕，备选音频转录
        """
        self.status_callback(f"🎥 Processing video / 处理视频: {url}")
        
        # Try to get subtitles
        content = self._get_subtitles(url)
        if content:
            self.status_callback("✅ Using video subtitles / 使用视频字幕")
            return Document(page_content=content, metadata={"source": url, "type": "subtitles"})
        
        # Fallback: audio transcription
        self.status_callback("⚠️ No subtitles found / 未找到字幕")
        
        if not allow_transcription:
            raise Exception("No subtitles found and audio transcription not allowed / 未找到字幕且不允许音频转录")
        
        self.status_callback("🎙️ Starting audio transcription... / 开始音频转录...")
        content = self._transcribe_audio(url)
        self.status_callback("✅ Using audio transcription / 使用音频转录")
        return Document(page_content=content, metadata={"source": url, "type": "transcription"})
    
    def _get_subtitles(self, url: str) -> Optional[str]:
        """Get YouTube subtitles using yt-dlp"""
        try:
            tmpdir = tempfile.mkdtemp()
            subtitle_path = os.path.join(tmpdir, "subtitle.%(ext)s")
            
            # Let yt-dlp choose the safest client; forcing ios/android now needs PO tokens.
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--write-auto-subs",
                "--sub-langs", "zh-Hans,zh,zh-CN,zh-TW,en,ja,ko,es,fr,de,pt,ru,ar,hi,it,nl,sv,no,da,fi,pl,tr,th,vi",
                "--skip-download",
                "--sub-format", "vtt/srt/best",
                "-o", subtitle_path,
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stderr:
                self.status_callback(f"yt-dlp stderr: {result.stderr[:200]}...")
            
            files = os.listdir(tmpdir)
            self.status_callback(f"📁 Files found: {files}")
            
            subtitle_files = [f for f in files if f.endswith(('.vtt', '.srt'))]
            
            if subtitle_files:
                subtitle_file = os.path.join(tmpdir, subtitle_files[0])
                self.status_callback(f"📄 Found subtitle file: {subtitle_files[0]}")
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                content = self._sanitize_text_for_storage(raw_content)
                
                # Cleanup
                for f in subtitle_files:
                    try:
                        os.remove(os.path.join(tmpdir, f))
                    except:
                        pass
                try:
                    os.rmdir(tmpdir)
                except:
                    pass
                
                return content
            
            try:
                os.rmdir(tmpdir)
            except:
                pass
            return None
            
        except Exception as e:
            self.status_callback(f"字幕提取异常: {str(e)}")
            return None
    
    def _transcribe_audio(self, url: str) -> str:
        """Download audio and transcribe using Whisper"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.m4a"
            
            # Avoid forcing ios/android clients; they now require PO tokens and fail without them.
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "-f", "bestaudio[ext=m4a]/bestaudio/best",
                "--no-progress",
                "-o", str(audio_path),
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                error_msg = stderr if stderr else stdout
                snippet = error_msg[:500] if error_msg else "unknown error"
                self.status_callback(f"yt-dlp audio download failed: {snippet}")
                raise RuntimeError(
                    "无法下载音频。可能是网络受限、视频受版权限制或需要登录。/ "
                    "Audio download failed. Please check network access or video restrictions.\n"
                    f"Details: {snippet}"
                )
            
            client = OpenAI()
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            
            return self._sanitize_text_for_storage(transcript.text)
    
    def generate_summary(self, document: Document, language: str = "en") -> str:
        """Generate video summary using the transcript"""
        self.status_callback("📝 Generating video summary... / 生成视频摘要...")
        
        try:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=1,
                model_kwargs={"max_completion_tokens": 2048}
            )
            cleaned_text = self._clean_subtitle_text(document.page_content)
            text = cleaned_text

            if not text:
                return "⚠️ Transcript contains no usable text / 转录内容为空"

            max_bytes = 500_000  # ~0.5MB safety margin well under provider cap
            truncated_notice = ""
            encoded = text.encode('utf-8')
            if len(encoded) > max_bytes:
                self.status_callback(
                    "✂️ Transcript very long; truncating to ~0.5MB to avoid API limits."
                )
                text = encoded[:max_bytes].decode('utf-8', errors='ignore')
                truncated_notice = "\n\n⚠️ Note: Transcript truncated to first ~0.5MB to avoid API limits."

            estimated_tokens = self._estimate_tokens(text)

            # Model-specific token limits
            model_limits = {
                "gpt-5-mini": 100000,
                "gpt-4o-mini": 100000,
                "gpt-4o": 100000,
                "gpt-4-turbo": 100000,
                "gpt-4": 6000,
                "gpt-3.5-turbo": 12000
            }
            
            max_tokens = model_limits.get(self.model_name, 6000)

            language = (language or "en").lower()
            if language not in {"zh", "en"}:
                language = "en"

            language_label = "Simplified Chinese" if language == "zh" else "English"
            language_instruction = "请使用简体中文撰写摘要。" if language == "zh" else "Please write the summary in English."
            language_reminder = "请用简体中文回答。" if language == "zh" else "Please respond in English."
            
            if estimated_tokens <= max_tokens:
                # Direct summarization for short texts
                summary_prompt = f"""You are a helpful assistant skilled at summarizing YouTube videos.  
I will provide you with the transcript of a video. Please create a concise summary that:  

1. Explains the core topic and main ideas of the video.  
2. Extracts the key points in a logical order, preferably as a list.  
3. Highlights any conclusions, recommendations, or methods mentioned.  
4. Avoids repeating sentences verbatim; paraphrase in your own words.  
5. Uses simple, clear language so someone who hasn't watched the video can understand.  
6. Writes the summary in {language_label}.  

Now summarize the following transcript:

{{transcript}}

{language_reminder}"""
                
                prompt = ChatPromptTemplate.from_template(summary_prompt)
                chain = prompt | llm
                result = chain.invoke({"transcript": text})
                summary = result.content + truncated_notice
                
            else:
                # Chunked summarization for long texts
                self.status_callback(f"📄 Text is long ({estimated_tokens} estimated tokens), using chunked summarization...")
                
                chunks = self._chunk_text_for_summary(text, max_tokens)
                chunk_summaries = []
                
                chunk_prompt = f"""Summarize this part of a YouTube video transcript. Focus on:
1. Main topics and key points
2. Important information and conclusions
3. Keep it concise (under 200 words) but comprehensive
4. Respond in {language_label}.

Transcript part:
{{chunk}}

{language_reminder}"""
                
                # Summarize each chunk
                for i, chunk in enumerate(chunks):
                    self.status_callback(f"📝 Summarizing chunk {i+1}/{len(chunks)}...")
                    chunk_template = ChatPromptTemplate.from_template(chunk_prompt)
                    chunk_chain = chunk_template | llm
                    chunk_result = chunk_chain.invoke({"chunk": chunk})
                    chunk_summary = chunk_result.content.strip()
                    if len(chunk_summary) > 1500:
                        chunk_summary = chunk_summary[:1500] + "..."
                    chunk_summaries.append(chunk_summary)
                
                # Combine chunk summaries
                combined_text = "\n\n".join(chunk_summaries)
                
                if self._estimate_tokens(combined_text) > max_tokens:
                    combined_text = combined_text[: max_tokens * 2]
                    combined_text += "\n\n(Chunk summaries truncated for final synthesis due to length.)"

                if len(combined_text) > 10000:
                    combined_text = combined_text[:10000] + "\n\n(Combined summary truncated to keep request small.)"
                
                final_prompt = f"""You are a helpful assistant skilled at summarizing YouTube videos.
I will provide you with summaries of different parts of a video transcript. Please create a comprehensive final summary that:

1. Explains the core topic and main ideas of the video.
2. Extracts the key points in a logical order, preferably as a list.
3. Highlights any conclusions, recommendations, or methods mentioned.
4. Avoids repeating information; synthesize and organize the content.
5. Uses simple, clear language so someone who hasn't watched the video can understand.
6. Writes the final summary in {language_label}.

Chunk summaries to synthesize:

{{chunk_summaries}}

{language_reminder}"""
                
                final_template = ChatPromptTemplate.from_template(final_prompt)
                final_chain = final_template | llm
                final_result = final_chain.invoke({"chunk_summaries": combined_text})
                summary = final_result.content + truncated_notice
            
            self.status_callback("✅ Summary generated successfully / 摘要生成成功")
            return summary
            
        except Exception as e:
            # Fallback: try with truncated text
            try:
                self.status_callback("🔄 Trying with truncated text... / 尝试使用截断文本...")
                truncated_text = cleaned_text[:2000]
                
                fallback_prompt = f"""You are a helpful assistant skilled at summarizing YouTube videos.
This is a truncated transcript (beginning portion) of a video. Please create a summary based on available content:

1. Explain what topics are covered in this portion
2. Extract key points mentioned
3. Note that this is a partial summary due to length constraints
4. Write the summary in {language_label}.

Transcript (truncated):
{{transcript}}

{language_reminder}"""
                
                prompt = ChatPromptTemplate.from_template(fallback_prompt)
                chain = prompt | llm
                result = chain.invoke({"transcript": truncated_text})
                
                summary = (
                    "⚠️ Partial Summary (video too long) / 部分摘要（视频过长）:\n\n"
                    + result.content
                    + truncated_notice
                )
                self.status_callback("✅ Fallback summary generated / 备用摘要生成成功")
                return summary
                
            except Exception:
                return "❌ Unable to generate summary due to text length limitations / 由于文本长度限制无法生成摘要"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 3
    
    def _chunk_text_for_summary(self, text: str, max_tokens: int = 12000) -> list:
        """Chunk text for summary generation"""
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return [text]

        # Keep a safety margin so each chunk stays well within the per-request limit.
        safe_tokens = max(1000, int(max_tokens * 0.75))
        max_chars = max(1000, safe_tokens)

        chunks = []
        current_chunk = []
        current_len = 0

        for line in text.splitlines():
            line_len = len(line) + 1  # account for newline

            if current_len + line_len > max_chars:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                if line_len >= max_chars:
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i:i + max_chars])
                    continue

            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _clean_subtitle_text(self, text: str) -> str:
        """Remove VTT/SRT timestamps and markup to keep transcript compact"""
        if not text:
            return text

        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('WEBVTT') or stripped.startswith('NOTE'):
                continue
            if '-->' in stripped:
                continue
            if re.match(r'^\d{1,4}$', stripped):
                continue
            # Remove HTML tags or leftover cue markers
            stripped = re.sub(r'<[^>]+>', '', stripped)
            stripped = stripped.strip()
            if stripped:
                cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines)

    def _sanitize_text_for_storage(self, text: str, max_bytes: int = 500_000) -> str:
        """Clean and truncate text so downstream requests stay within limits"""
        if not text:
            return ""

        cleaned = self._clean_subtitle_text(text)
        if not cleaned:
            return ""

        encoded = cleaned.encode('utf-8')
        if len(encoded) <= max_bytes:
            return cleaned

        truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
        return truncated
    
    def save_summary(self, summary: str, document: Document, save_to_file: bool = False) -> Optional[str]:
        """Save summary to file if requested"""
        if not summary or not save_to_file:
            return None
            
        source_url = document.metadata.get("source", "unknown")
        video_id = extract_video_id(source_url)
        filename = f"{video_id}_summary.txt"
        
        header = f"Video Summary / 视频摘要\nSource: {source_url}"
        return save_text_file(summary, filename, header)
    
    def save_original_text(self, document: Document, save_to_file: bool = False) -> Optional[str]:
        """Save original text to file if requested"""
        if not save_to_file:
            return None
            
        content_type = document.metadata.get("type", "content")
        source_url = document.metadata.get("source", "unknown")
        video_id = extract_video_id(source_url)
        filename = f"{video_id}_{content_type}.txt"
        
        header = f"Source: {source_url}\nType: {content_type}"
        return save_text_file(document.page_content, filename, header)

    def generate_analysis_report(
        self,
        document: Document,
        summary: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> str:
        """Generate a structured analysis report based on transcript and summary"""
        self.status_callback("🧠 Generating detailed analysis... / 生成详细分析...")

        try:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                model_kwargs={"max_completion_tokens": 2048}
            )

            language = (language or "en").lower()
            if language not in {"zh", "en"}:
                language = "en"

            language_instruction = "请使用简体中文撰写分析。" if language == "zh" else "Please write the analysis in English."
            language_reminder = "请用简体中文回答。" if language == "zh" else "Please respond in English."

            cleaned_text = self._clean_subtitle_text(document.page_content)
            transcript_excerpt = self._sanitize_text_for_storage(cleaned_text, max_bytes=120_000)
            if not transcript_excerpt:
                transcript_excerpt = (summary or "")[:4000]
            elif len(transcript_excerpt) > 20_000:
                transcript_excerpt = transcript_excerpt[:20_000]

            context_metadata = []
            metadata = metadata or {}
            video_urls = metadata.get("video_urls") or ([metadata.get("video_url")] if metadata.get("video_url") else [])
            if video_urls:
                context_metadata.append("Video URLs: " + ", ".join(video_urls))
            if metadata.get("created_at"):
                context_metadata.append(f"Session Created At: {metadata['created_at']}")
            if metadata.get("model_name"):
                context_metadata.append(f"Model: {metadata['model_name']}")
            if metadata.get("chunk_size"):
                context_metadata.append(f"Chunk Size: {metadata['chunk_size']}")

            metadata_block = "\n".join(context_metadata) if context_metadata else "(No additional metadata provided)"

            analysis_prompt = f"""You are an investment analyst specializing in Warren Buffett and Berkshire Hathaway annual shareholder meetings. 
Using the provided summary, transcript excerpt, and metadata, craft a structured analysis of the meeting. {language_instruction}

The analysis must be returned as Markdown with the following sections (use headings with '## '):
1. Overview — concise description of the meeting and its context.
2. Key Themes — bullet list summarizing the major discussion topics and Buffett's perspectives.
3. Investor Takeaways — actionable lessons or strategic insights relevant to long-term investors.
4. Q&A Highlights — summarize notable questions from shareholders and Buffett/Charlie’s responses.
5. Notable Quotes — include 3-5 memorable quotes with approximate timestamps if evident in the excerpt; if no timestamps are available, note that.
6. Suggested Follow-up Questions — list questions worth exploring in future research or subsequent meetings.

Keep the tone analytical yet accessible. Reference concrete details from the context when possible. Avoid inventing facts that are not supported by the provided materials.

Summary:
{{summary_block}}

Transcript Excerpt:
{{transcript_block}}

Metadata:
{{metadata_block}}

{language_reminder}"""

            prompt = ChatPromptTemplate.from_template(analysis_prompt)
            chain = prompt | llm
            result = chain.invoke({
                "summary_block": summary or "(Summary not available)",
                "transcript_block": transcript_excerpt or "(Transcript excerpt not available)",
                "metadata_block": metadata_block
            })

            return result.content.strip()

        except Exception as exc:
            self.status_callback(f"❌ Analysis generation failed: {exc}")
            raise
