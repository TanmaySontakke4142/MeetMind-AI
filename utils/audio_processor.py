import glob
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = str(ROOT / "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mkv", ".webm", ".mov", ".wma", ".mpeg", ".mpga", ".m4v",
}


def ffmpeg_bin_dir() -> str:
    """Find ffmpeg on PATH, via FFMPEG_PATH, or in the usual Windows install locations."""
    found = shutil.which("ffmpeg")
    if found:
        return os.path.dirname(found)

    override = (os.getenv("FFMPEG_PATH") or "").strip().strip('"')
    if override:
        path = Path(override)
        if path.is_file():
            return str(path.parent)
        if (path / "ffmpeg.exe").is_file() or (path / "ffmpeg").is_file():
            return str(path)

    local = os.environ.get("LOCALAPPDATA", "")
    patterns = [
        os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe"),
        os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])

    raise RuntimeError(
        "FFmpeg was not found. Install it with 'winget install Gyan.FFmpeg', then restart the terminal, "
        "or set FFMPEG_PATH to the folder that contains ffmpeg.exe."
    )


def _prefer_ffmpeg_on_path() -> None:
    try:
        bindir = ffmpeg_bin_dir()
    except RuntimeError:
        return
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


_prefer_ffmpeg_on_path()

import yt_dlp  # noqa: E402
from pydub import AudioSegment  # noqa: E402


def ensure_ffmpeg() -> str:
    bindir = ffmpeg_bin_dir()
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    AudioSegment.converter = os.path.join(bindir, exe)
    AudioSegment.ffprobe = os.path.join(bindir, probe)
    return bindir


def download_youtube_audio(url: str) -> str:
    bindir = ensure_ffmpeg()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "windowsfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": bindir,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("The site returned no video information.")
            prepared = ydl.prepare_filename(info)
    except Exception as exc:
        raise RuntimeError(
            "Could not download audio from that link. Use a public YouTube URL. "
            f"Details: {exc}"
        ) from exc

    wav_path = os.path.splitext(prepared)[0] + ".wav"
    if not os.path.isfile(wav_path):
        raise RuntimeError(
            "The download finished, but the WAV file was not created. Check that FFmpeg works."
        )
    return wav_path


def convert_to_wav(input_path: str) -> str:
    ensure_ffmpeg()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    extension = os.path.splitext(input_path)[1].lower()
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported file type '{extension}'. Upload audio or video such as mp3, wav, m4a, mp4, or webm."
        )

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not read this file. It may be corrupt or in an unsupported format. "
            f"Details: {exc}"
        ) from exc

    if len(audio) < 200:
        raise RuntimeError("This file has almost no audio to transcribe.")

    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    if not chunks:
        raise RuntimeError("No audio chunks could be created from this file.")
    return chunks


def process_input(source: str) -> list:
    source = (source or "").strip().strip('"')
    if not source:
        raise ValueError("Provide a YouTube link or a local audio/video file.")

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected a URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected a local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
