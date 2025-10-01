#!/usr/bin/env python3.12
"""Batch ingest a YouTube playlist into a single RAG session."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Set

from dotenv import load_dotenv


def _init_paths() -> None:
    """Ensure the project src directory is importable."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


load_dotenv()
_init_paths()

from yt_dlp import YoutubeDL  # noqa: E402  pylint: disable=wrong-import-position

from youtube_rag_system import YouTubeRAG  # noqa: E402  pylint: disable=wrong-import-position


def fetch_playlist_entries(url: str) -> List[Dict[str, Any]]:
    """Return a flat list of video entries for the provided playlist URL."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        raise ValueError("Unable to retrieve playlist metadata")

    entries = info.get("entries")
    if not entries:
        # Some URLs (e.g., single videos) may not expose entries.
        if info.get("webpage_url"):
            return [info]
        raise ValueError("The provided URL does not contain any videos")

    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if not entry:
            continue
        video_url = entry.get("url") or entry.get("webpage_url") or ""
        if video_url and not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={video_url}"

        normalized.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title") or "(untitled)",
                "url": video_url,
            }
        )

    filtered = [item for item in normalized if item["url"]]
    if not filtered:
        raise ValueError("Playlist contains entries without usable URLs")

    return filtered


def iter_limited(entries: List[Dict[str, Any]], start: int = 0, limit: int | None = None) -> Iterable[Dict[str, Any]]:
    """Yield a slice of playlist entries based on start/limit controls."""
    selected = entries[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube playlist into a single YouTubeRAG session."
    )
    parser.add_argument("playlist", help="YouTube playlist or channel URL")
    parser.add_argument(
        "--session-name",
        help="Existing session name to append to; otherwise a new session is created",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI chat model used for summarisation (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Chunk overlap for text splitting (default: 20)",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Language for summaries and analyses (default: zh)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based index to start ingesting from",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Limit number of videos to ingest (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos already present in the session metadata",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Persist per-video summary files alongside the session",
    )
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="Persist cleaned transcript text files alongside the session",
    )
    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable Whisper fallback when subtitles are unavailable",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort ingestion when a single video fails (default: continue)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate status messages from the RAG engine",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    status_printer = print if args.verbose else (lambda *_: None)
    rag = YouTubeRAG(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        language=args.language,
        status_callback=status_printer,
    )

    try:
        playlist_entries = fetch_playlist_entries(args.playlist)
    except Exception as exc:  # pylint: disable=broad-except
        parser.error(f"Failed to resolve playlist: {exc}")

    entries_iter = iter_limited(
        playlist_entries,
        start=max(args.start_index, 0),
        limit=args.max_videos,
    )

    session_data: Dict[str, Any] | None = None
    session_name = args.session_name
    processed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    if session_name:
        session_data = rag.load_session(session_name)
        if not session_data:
            parser.error(f"Unable to load session '{session_name}'")

    known_urls: Set[str] = set()
    if session_data:
        known_urls.update(session_data.get("video_urls", []))

    allow_transcription = not args.no_transcription

    for entry in entries_iter:
        video_url = entry.get("url")
        title = entry.get("title") or entry.get("id") or "(unknown title)"
        if not video_url:
            failed.append(entry)
            continue

        if args.skip_existing and session_data and video_url in known_urls:
            print(f"⏭️  Skipping already ingested video: {title}")
            continue

        print(f"🎬 Processing: {title}\n    URL: {video_url}")
        try:
            if session_data is None:
                result = rag.process_video(
                    video_url,
                    allow_transcription=allow_transcription,
                    save_summary=args.save_summary,
                    save_original=args.save_original,
                )
                session_data = result
                session_name = result.get("session_name") or result.get("metadata", {}).get("persist_name")
                if not session_name:
                    raise ValueError("Session name missing after initial ingestion")
                known_urls.update(session_data.get("video_urls", []))
                print(f"✅ Created session '{session_name}'")
            else:
                result = rag.add_video_to_session(
                    session_data,
                    video_url,
                    allow_transcription=allow_transcription,
                )
                session_data = result["session"]
                known_urls.update(session_data.get("video_urls", []))
                print("✅ Appended video to session")

            processed.append(entry)

            if args.save_summary and result.get("new_summary"):
                latest_doc = session_data.get("documents", [])[-1]
                rag.content_processor.save_summary(result["new_summary"], latest_doc, save_to_file=True)

            if args.save_original:
                latest_doc = session_data.get("documents", [])[-1]
                rag.content_processor.save_original_text(latest_doc, save_to_file=True)

        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Failed to ingest '{title}': {exc}")
            failed.append(entry)
            if args.stop_on_error:
                break

    if session_name:
        print(f"\n📦 Final session: {session_name}")
        print(f"   Videos ingested: {len(processed)}")
        print(f"   Stored at: rag_sessions/{session_name}")

    if failed:
        print(f"\n⚠️  {len(failed)} videos failed. Review logs above for details.")


if __name__ == "__main__":
    main()
