import os
import tempfile

import whisper
from pydub import AudioSegment

from utils.audio_processor import ensure_ffmpeg

SARVAM_PIECE_SECONDS = 25
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

WHISPER_CHUNK_SECONDS = 30

_model = None


def load_model():
    global _model

    if _model is None:
        ensure_ffmpeg()
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")

        try:
            _model = whisper.load_model(WHISPER_MODEL)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Whisper model '{WHISPER_MODEL}'. Details: {exc}"
            ) from exc

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str, task: str = "transcribe") -> str:
    model = load_model()

    try:
        audio = AudioSegment.from_file(chunk_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        if len(audio) == 0:
            raise RuntimeError("The audio file is empty.")

        results = []

        chunk_ms = WHISPER_CHUNK_SECONDS * 1000

        for start in range(0, len(audio), chunk_ms):
            segment = audio[start:start + chunk_ms]

            if len(segment) == 0:
                continue

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                segment_path = temp_file.name

            try:
                segment.export(segment_path, format="wav")

                result = model.transcribe(
                    segment_path,
                    task=task,
                    fp16=False,
                    verbose=False,
                )

                text = (result or {}).get("text", "").strip()

                if text:
                    results.append(text)

            finally:
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        return " ".join(results).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Whisper could not transcribe {os.path.basename(chunk_path)}. "
            f"Details: {exc}"
        ) from exc


def _send_to_sarvam(piece_path: str) -> str:
    import requests

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as handle:
        files = {
            "file": (os.path.basename(piece_path), handle, "audio/wav")
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        try:
            response = requests.post(
                SARVAM_STT_TRANSLATE_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach Sarvam AI. Details: {exc}"
            ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Sarvam AI returned {response.status_code}. "
            "Check your SARVAM_API_KEY."
        )

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set.")

    audio = AudioSegment.from_file(chunk_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = []

    total_pieces = max(
        1, (len(audio) + piece_ms - 1) // piece_ms
    )

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]

        if len(piece) == 0:
            continue

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_file:
            piece_path = temp_file.name

        try:
            piece.export(piece_path, format="wav")

            print(f"-> Sarvam piece {i + 1}/{total_pieces}")

            text = _send_to_sarvam(piece_path)

            if text:
                full_text.append(text)

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(full_text).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    English: local Whisper transcription.
    Hinglish: Sarvam translation if API key exists,
    otherwise local Whisper translation to English.
    """

    if language.lower() == "hinglish" and SARVAM_API_KEY:
        return transcribe_chunk_sarvam(chunk_path)

    if language.lower() == "hinglish":
        return transcribe_chunk_whisper(chunk_path, task="translate")

    return transcribe_chunk_whisper(chunk_path, task="transcribe")


def transcribe_all(chunks: list, language: str = "english", on_progress=None) -> str:

    if not chunks:
        raise RuntimeError(
            "No audio chunks were produced, so there is nothing to transcribe."
        )

    if language.lower() == "hinglish" and SARVAM_API_KEY:
        engine = "Sarvam AI"

    elif language.lower() == "hinglish":
        engine = "Whisper (translate to English)"

    else:
        engine = "Whisper"

    print(f"Using {engine} for transcription.")

    parts = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):

        if on_progress:
            on_progress(i + 1, total)

        print(f"Transcribing chunk {i + 1}/{total}...")

        text = transcribe_chunk(chunk, language=language).strip()

        if text:
            parts.append(text)

    print("Transcription complete.")

    return " ".join(parts).strip()
