import hashlib
import html
import math
import os
import re
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_discussion_points,
    extract_key_decisions,
    extract_questions,
)
from core.llm import require_api_key
from core.rag_engine import build_rag_chain, ask_question

APP_NAME = "MeetMind AI"

st.set_page_config(
    page_title=f"{APP_NAME} - AI meeting assistant",
    page_icon="🎙️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#090d17; --panel:#111827; --panel2:#151e30; --text:#edf2ff; --muted:#9aa8c3; --purple:#8b5cf6; --teal:#2dd4bf; --line:rgba(148,163,184,.16); }
.stApp { background:radial-gradient(ellipse at 15% 0%, #202447 0%, transparent 38%), radial-gradient(ellipse at 95% 12%, #12363a 0%, transparent 30%), var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; }
.stApp :is(p,li,label,h1,h2,h3,h4,span) { color:var(--text); }
.block-container { max-width:1250px; padding-top:2rem; padding-bottom:4rem; }
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"] { visibility:hidden; } header[data-testid="stHeader"] { background:transparent; }
section[data-testid="stSidebar"] { background:#0b1020; border-right:1px solid var(--line); } section[data-testid="stSidebar"] :is(p,span,label,div) { color:var(--text); }
.brand { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; letter-spacing:-.04em; } .brand small { display:block; color:var(--muted); font-size:.82rem; font-weight:400; margin-top:.35rem; letter-spacing:0; }
.side-note { color:var(--muted); font-size:.9rem; line-height:1.6; padding:1rem 0; } .side-rule { border-top:1px solid var(--line); margin:1.2rem 0; }
.stButton button,.stDownloadButton button { border-radius:12px!important; border:1px solid var(--line)!important; background:linear-gradient(135deg,#182239,#121a2a)!important; color:var(--text)!important; font-weight:600!important; transition:all .18s ease!important; padding:.65rem 1rem!important; }
.stButton button:hover,.stDownloadButton button:hover { border-color:var(--purple)!important; transform:translateY(-1px); box-shadow:0 5px 20px rgba(139,92,246,.15); }
.stButton button[data-testid="stBaseButton-primary"],.stButton button[kind="primary"] { background:linear-gradient(135deg,#8b5cf6,#5b5bd6)!important; border-color:#8b5cf6!important; }
.stTextInput input,.stTextArea textarea { background:#101827!important; border:1px solid var(--line)!important; border-radius:12px!important; color:var(--text)!important; }
[data-testid="stFileUploaderDropzone"] { background:#101827; border:1px dashed #64748b; border-radius:16px; } [data-testid="stFileUploaderDropzone"] :is(span,small,p,div) { color:var(--text); }
[data-testid="stFileUploaderDropzone"] button { background:#182239!important; color:var(--text)!important; }
.stRadio label p { color:var(--text)!important; } .stTabs [data-baseweb="tab-list"] { gap:.5rem; border-bottom:1px solid var(--line); } .stTabs button[data-baseweb="tab"] { color:var(--muted); padding:.7rem; } .stTabs button[aria-selected="true"] { color:var(--text); border-bottom-color:var(--purple); }
.stTabs [data-testid="stMarkdownContainer"] :is(p,li) { line-height:1.75; } [data-testid="stChatMessage"] { background:#111827; border:1px solid var(--line); border-radius:16px; } [data-testid="stChatInput"] textarea { background:#101827!important; color:var(--text)!important; }
.stCheckbox { padding:.35rem 0; } .stCheckbox label { color:var(--text)!important; } .stCaption,[data-testid="stCaptionContainer"] { color:var(--muted); }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.1rem,4vw,3.6rem); font-weight:700; letter-spacing:-.045em; line-height:1.1; margin-bottom:.8rem; } .hero-copy { color:var(--muted); font-size:1.1rem; line-height:1.7; max-width:65ch; }
.card { background:linear-gradient(145deg,rgba(25,35,55,.94),rgba(15,21,34,.94)); border:1px solid var(--line); border-radius:20px; padding:1.25rem; margin:.3rem 0; } .eyebrow { color:#a78bfa; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; } .gets { border-left:2px solid var(--purple); padding-left:1.2rem; } .gets-row { padding:.8rem 0; border-bottom:1px solid var(--line); } .gets-row b { display:block; font-size:1.02rem; } .gets-row span { color:var(--muted); font-size:.92rem; }
.brief-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(1.7rem,3vw,2.5rem); font-weight:700; letter-spacing:-.03em; } .brief-meta,.tab-lead { color:var(--muted); } .brief-strip { line-height:1.7; } table.sheet { width:100%; border-collapse:collapse; } table.sheet td,table.sheet th { padding:.7rem; border-bottom:1px solid var(--line); text-align:left; } .note { padding:.8rem 0; border-bottom:1px solid var(--line); } .note .body { line-height:1.7; } .notebook { background:#101827; border:1px solid var(--line); border-radius:14px; padding:1rem; max-height:560px; overflow-y:auto; line-height:1.8; }
"""


def inject_css() -> None:
    st.markdown("<style>" + re.sub(r"\n\s*\n", "\n", CSS) + "</style>", unsafe_allow_html=True)


def _h(markup: str) -> None:
    """Render an HTML snippet. Newlines are collapsed so Markdown can't mangle it."""
    st.markdown(re.sub(r"\n\s*", "", markup.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)")


def action_fields(item: str) -> tuple[str, str, str]:
    """Split 'Task: ... | Assigned to: ... | Deadline: ...' into three columns."""
    task, owner, deadline = item.strip(), "Not specified", "Not specified"
    recognized = False
    for piece in re.split(r"\s*\|\s*", item):
        label, sep, value = piece.partition(":")
        if not sep:
            continue
        key = label.strip().lower()
        cleaned = value.strip() or "Not specified"
        if key == "task":
            task, recognized = cleaned, True
        elif "assign" in key or key in {"owner", "person", "who"}:
            owner, recognized = cleaned, True
        elif "deadline" in key or key in {"due", "when"}:
            deadline, recognized = cleaned, True
    if not recognized:
        task = item.strip()
    return task, owner, deadline


def parse_items(value) -> list[str]:
    """Turn LLM output (markdown bullets, numbered list, or a Python list) into clean items."""
    if value is None:
        return []
    lines = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).splitlines()
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _BULLET.match(line)
        if match:
            items.append(match.group(1).strip())
        elif line.endswith(":"):
            continue  # intro line such as "Here are the action items:"
        elif items:
            items[-1] += " " + line  # wrapped continuation of the previous bullet
        else:
            items.append(line)
    return items


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {v}" for v in value)
    return str(value)


def inline_md(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "meeting-brief"


def short_source(source: str) -> str:
    cleaned = re.sub(r"^https?://(www\.)?", "", source)
    return cleaned if len(cleaned) <= 60 else cleaned[:57] + "..."


def trace_svg(width: int = 1000, height: int = 56, n: int = 180) -> str:
    """A pen-trace of a voice: quiet, a burst of speech, quiet again."""
    points = []
    for i in range(n + 1):
        x = i / n * width
        envelope = math.sin(math.pi * i / n) ** 1.4
        wobble = 0.6 * math.sin(i * 0.9) + 0.4 * math.sin(i * 2.3 + 1)
        y = height / 2 + envelope * height * 0.44 * wobble
        points.append(f"{x:.1f},{y:.1f}")
    d = "M" + " L".join(points)
    return (
        f'<svg class="trace" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
        f'<path d="{d}" pathLength="1" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>'
    )


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

def build_txt(r: dict) -> str:
    def block(head: str, body: str) -> str:
        return f"{head}\n{'-' * len(head)}\n{body.strip() or 'None found.'}\n"

    return "\n".join(
        [
            r["title"],
            "=" * len(r["title"]),
            f"Source: {r['source_label']}",
            f"Generated: {r['generated']}",
            "",
            block("Summary", as_text(r["summary"])),
            block("Action items", as_text(r["action_items"])),
            block("Key decisions", as_text(r["key_decisions"])),
            block("Discussion points", as_text(r.get("discussion_points"))),
            block("Open questions", as_text(r["open_questions"])),
            block("Transcript", r["transcript"]),
        ]
    )


_PDF_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...",
    "\u00a0": " ", "\u2192": "->",
}


def pdf_safe(text) -> str:
    """The built-in PDF fonts only cover Latin-1, so map or replace anything outside it."""
    text = str(text)
    for old, new in _PDF_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(r: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    nxt = {"new_x": "LMARGIN", "new_y": "NEXT"}

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(22, 37, 29)
    pdf.multi_cell(0, 9, pdf_safe(r["title"]), **nxt)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(95, 110, 100)
    pdf.multi_cell(0, 5, pdf_safe(f"Source: {r['source_label']}   Generated: {r['generated']}"), **nxt)
    pdf.ln(2)

    def section(head: str, body: str) -> None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(22, 37, 29)
        pdf.cell(0, 8, head, **nxt)
        pdf.set_draw_color(198, 61, 47)
        pdf.set_line_width(0.6)
        y = pdf.get_y()
        pdf.line(18, y, 44, y)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(40, 52, 46)
        pdf.multi_cell(0, 5.8, pdf_safe(body).strip() or "None found.", **nxt)

    section("Summary", as_text(r["summary"]))
    section("Action items", as_text(r["action_items"]))
    section("Key decisions", as_text(r["key_decisions"]))
    section("Discussion points", as_text(r.get("discussion_points")))
    section("Open questions", as_text(r["open_questions"]))
    section("Transcript", r["transcript"])
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Pipeline (same steps as run_pipeline, with progress shown on screen)
# --------------------------------------------------------------------------

def run_with_progress(source: str | None, language: str, source_label: str, cache_key: tuple | None = None):
    started = time.time()
    box = st.status("Working through the recording", expanded=True)
    try:
        box.write("Checking the Gemini API key")
        print("[Pipeline] Stage 1: Checking Gemini API key")
        require_api_key()

        cached = st.session_state.get("cached_transcript")
        if cache_key and cached and cached.get("cache_key") == cache_key and cached.get("transcript"):
            box.write("Using the saved transcript from the previous attempt (skipping Whisper)")
            transcript = cached["transcript"]
            print(f"[Pipeline] Stage 2-3: Reusing cached transcript (chars={len(transcript)}, language={language})")
        else:
            box.write("Preparing the audio")
            print(f"[Pipeline] Stage 2: Audio preparation started (language={language})")
            chunks = process_input(source)
            print(f"[Pipeline] Stage 2: Audio preparation completed (chunks={len(chunks)})")

            box.write("Transcribing with Whisper on this computer. This is the slowest step.")
            print(f"[Pipeline] Stage 3: Transcription started (chunks={len(chunks)}, language={language})")

            def on_progress(index: int, total: int) -> None:
                box.write(f"Transcribing chunk {index} of {total}")

            transcript = transcribe_all(chunks, language=language, on_progress=on_progress)
            if not transcript or not transcript.strip():
                raise ValueError("No speech was found in this recording.")

            print(f"[Pipeline] Stage 3: Transcription completed (chars={len(transcript)}, words={len(transcript.split())})")
            st.session_state.cached_transcript = {
                "cache_key": cache_key,
                "transcript": transcript,
                "source_label": source_label,
                "language": language,
                "failed": False,
            }

        box.write("Writing the title and summary")
        print("[Pipeline] Stage 4: Invoking Gemini for title and summary")
        title = generate_title(transcript)
        summary = summarize(transcript)
        print(f"[Pipeline] Stage 4: Title & summary completed (title_chars={len(str(title))}, summary_chars={len(str(summary))})")

        box.write("Extracting action items, decisions, discussion points and open questions")
        print("[Pipeline] Stage 5: Invoking Gemini for structured extractions")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        discussion = extract_discussion_points(transcript)
        questions = extract_questions(transcript)
        print(
            "[Pipeline] Stage 5: Extractions completed "
            f"(actions_chars={len(str(action_items))}, decisions_chars={len(str(decisions))}, "
            f"discussion_chars={len(str(discussion))}, questions_chars={len(str(questions))})"
        )

        box.write("Indexing the transcript locally with MiniLM and ChromaDB")
        print("[Pipeline] Stage 6: Building ChromaDB vector store & RAG chain")
        rag_chain = build_rag_chain(transcript)
        print("[Pipeline] Stage 6: RAG chain ready")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the person
        print(f"[Pipeline] Error: {type(exc).__name__}: {exc}")
        if st.session_state.get("cached_transcript"):
            st.session_state.cached_transcript["failed"] = True
        box.update(label="Could not finish", state="error", expanded=True)
        st.error(str(exc))
        return None

    if st.session_state.get("cached_transcript"):
        st.session_state.cached_transcript["failed"] = False
    elapsed = int(time.time() - started)
    box.update(label=f"Done in {elapsed // 60}m {elapsed % 60}s", state="complete", expanded=False)

    result = {
        "rid": int(time.time() * 1000),
        "title": str(title).strip().strip('"'),
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "discussion_points": discussion,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "source_label": source_label,
        "language": language,
        "generated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    result["txt"] = build_txt(result)
    try:
        result["pdf"] = build_pdf(result)
    except Exception:  # noqa: BLE001 - PDF is optional; TXT still works
        result["pdf"] = None
    print(f"[State] Built result rid={result['rid']} (has_rag={result['rag_chain'] is not None})")
    return result


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def render_sidebar() -> None:
    result = st.session_state.result
    with st.sidebar:
        _h(f'<div class="brand">✦ {APP_NAME}<small>Turn conversations into action</small></div>')
        st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
        if st.button("⌂   Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.page = "Dashboard"; st.rerun()
        if st.button("◉   Meeting Studio", use_container_width=True, key="nav_studio"):
            st.session_state.page = "Meeting Studio"; st.rerun()
        if st.button("✦   Meeting Workspace", use_container_width=True, key="nav_workspace", disabled=result is None):
            st.session_state.page = "Meeting Workspace"; st.rerun()
        st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
        if result is not None:
            slug = slugify(result["title"])
            if result.get("pdf"):
                st.download_button("↓  Download PDF", data=result["pdf"], file_name=f"{slug}.pdf", mime="application/pdf", type="primary", key="dl_pdf")
            st.download_button("↓  Download TXT", data=result["txt"], file_name=f"{slug}.txt", mime="text/plain", key="dl_txt")
            if st.button("＋  Start a new meeting", use_container_width=True, key="reset"):
                st.session_state.result = None; st.session_state.chat = []; st.session_state.cached_transcript = None; st.session_state.page = "Meeting Studio"; st.rerun()
        else:
            _h('<div class="side-note">Upload a recording or add a video link to create a meeting brief, extract tasks and explore the transcript.</div>')


def render_dashboard() -> None:
    _h('<div class="eyebrow">YOUR AI MEETING WORKSPACE</div><div class="hero-title">Turn conversations<br>into <span style="color:#a78bfa">action.</span></div><p class="hero-copy">Capture the important moments. Find decisions, follow-ups and answers without replaying the entire meeting.</p>')
    c1,c2,c3=st.columns(3)
    for col,icon,title,desc in [(c1,"◉","Meeting Studio","Upload audio, video or a YouTube link."),(c2,"✦","AI insights","Summaries, decisions and tasks."),(c3,"⌕","Meeting chat","Ask questions grounded in your transcript.")]:
        with col: _h(f'<div class="card"><div style="font-size:1.8rem;color:#a78bfa">{icon}</div><h3>{title}</h3><p style="color:#9aa8c3">{desc}</p></div>')
    st.markdown("### Start here")
    a,b=st.columns([1,1])
    with a:
        if st.button("＋  Create a new meeting", type="primary", use_container_width=True, key="dash_new"):
            st.session_state.page="Meeting Studio"; st.rerun()
    with b:
        if st.button("◉  Open meeting workspace", use_container_width=True, key="dash_workspace", disabled=st.session_state.result is None):
            st.session_state.page="Meeting Workspace"; st.rerun()
    if st.session_state.result:
        r=st.session_state.result
        _h('<div class="card"><div class="eyebrow">LATEST MEETING</div><h3>'+html.escape(r.get("title","Meeting"))+'</h3><p style="color:#9aa8c3">'+html.escape(r.get("source_label",""))+'</p></div>')
        if st.button("View latest meeting →", type="primary", key="dash_latest"):
            st.session_state.page="Meeting Workspace"; st.rerun()
    else:
        st.info("Your meeting library will appear here after you analyze a recording. No sample or placeholder meeting data is shown.")


def render_landing() -> None:
    _h(
        '<div class="hero-title">Hand over the recording. Get back the decisions.</div>'
        '<p class="hero-copy">Paste a YouTube link or upload a meeting recording. Whisper transcribes it '
        "on your machine, then you get a summary, action items, decisions, discussion points "
        "and a chat that answers from the transcript.</p>"
        + trace_svg()
    )

    form, gets = st.columns([1.25, 1], gap="large")

    with form:
        _h('<div class="form-title">Add a recording</div>')
        mode = st.radio(
            "Source", ["YouTube link", "Upload a file"],
            horizontal=True, label_visibility="collapsed", key="mode",
        )
        url, upload = "", None
        if mode == "YouTube link":
            url = st.text_input(
                "YouTube link", placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed", key="url",
            ).strip()
        else:
            upload = st.file_uploader(
                "Recording", type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "mkv", "webm", "mov"],
                label_visibility="collapsed", key="upload",
            )

        language = st.radio(
            "Spoken language", ["english", "hinglish"],
            format_func=str.title, horizontal=True, key="language",
        )
        st.caption(
            "Hinglish is translated to English on this computer with Whisper. "
            "If SARVAM_API_KEY is set, that service is used instead."
        )
        go = st.button("Generate brief", type="primary", key="go")

        if go:
            source, label, cache_key = None, "", None
            cached = st.session_state.get("cached_transcript")
            if mode == "YouTube link":
                if not url:
                    st.warning("Paste a YouTube link first.")
                else:
                    source, label = url, short_source(url)
                    cache_key = ("url", url, language)
            elif upload is None:
                st.warning("Choose a recording to upload first.")
            else:
                buf = upload.getbuffer()
                sample = bytes(buf[:65536]) + (bytes(buf[-65536:]) if len(buf) > 65536 else b"")
                digest = hashlib.sha256(sample).hexdigest()
                cache_key = ("file", upload.name, upload.size, digest, language)
                label = upload.name
                if cached and cached.get("cache_key") == cache_key and cached.get("transcript"):
                    source = "cached"
                else:
                    folder = tempfile.mkdtemp(prefix="minutes_")
                    source = os.path.join(folder, os.path.basename(upload.name))
                    with open(source, "wb") as fh:
                        fh.write(buf)

            if source:
                result = run_with_progress(source, language, label, cache_key=cache_key)
                if result:
                    st.session_state.result = result
                    st.session_state.chat = []
                    st.session_state.page = "Meeting Workspace"
                    print(f"[State] Saved result rid={result['rid']} and switching page to 'Meeting Workspace'")
                    st.rerun()

        cached = st.session_state.get("cached_transcript")
        if cached and cached.get("failed") and cached.get("transcript"):
            st.info("Your transcript was saved so you will not have to transcribe this recording again.")
            retry_col, dl_col = st.columns(2)
            with retry_col:
                if st.button("Retry AI analysis", type="primary", key="retry_ai"):
                    result = run_with_progress(
                        None,
                        cached["language"],
                        cached["source_label"],
                        cache_key=cached["cache_key"],
                    )
                    if result:
                        st.session_state.result = result
                        st.session_state.chat = []
                        st.session_state.page = "Meeting Workspace"
                        print(f"[State] Retry succeeded rid={result['rid']} and switching page to 'Meeting Workspace'")
                        st.rerun()
            with dl_col:
                st.download_button(
                    "↓  Download transcript",
                    data=cached["transcript"],
                    file_name="transcript.txt",
                    mime="text/plain",
                    key="dl_cached_tx",
                )

    with gets:
        _h(
            '<div class="gets">'
            '<div class="gets-row"><b>Summary</b><span>The meeting in a few short paragraphs.</span></div>'
            '<div class="gets-row"><b>Action items</b><span>A list you can tick off as work gets done.</span></div>'
            '<div class="gets-row"><b>Decisions</b><span>What the group agreed on.</span></div>'
            '<div class="gets-row"><b>Discussion</b><span>The main points that were talked through.</span></div>'
            '<div class="gets-row"><b>Open questions</b><span>What is still unresolved.</span></div>'
            '<div class="gets-row"><b>Transcript</b><span>Full text with keyword search.</span></div>'
            '<div class="gets-row"><b>Chat</b><span>Ask anything and get answers from the transcript.</span></div>'
            "</div>"
        )


def render_notes(value, kind: str, glyph: str, empty: str) -> None:
    items = parse_items(value)
    if not items:
        st.markdown(f'<p class="tab-lead">{empty}</p>', unsafe_allow_html=True)
        return
    _h("".join(
        f'<div class="note {kind}"><span class="glyph">{glyph}</span>'
        f'<span class="body">{inline_md(item)}</span></div>'
        for item in items
    ))


def render_transcript(text: str) -> None:
    query = st.text_input(
        "Search the transcript", placeholder="Search the transcript",
        label_visibility="collapsed", key="tx_search",
    ).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    paragraphs = [" ".join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]

    matches = 0
    out = []
    for para in paragraphs:
        safe = html.escape(para)
        if query:
            pattern = re.compile(re.escape(html.escape(query)), re.I)
            matches += len(pattern.findall(safe))
            safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
        out.append(f"<p>{safe}</p>")

    if query:
        st.caption(f"{plural(matches, 'match')} for \"{query}\"" if matches else f"No matches for \"{query}\".")
    _h(f'<div class="notebook">{"".join(out)}</div>')


def render_chat(result: dict) -> None:
    history = st.session_state.chat
    suggestions = [
        "What were the main topics?",
        "Who is responsible for what?",
        "What was decided, and why?",
    ]

    clicked = None
    if not history:
        st.markdown(
            '<p class="tab-lead">Ask about anything said in the recording. '
            "Answers come from the transcript only.</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(suggestions))
        for col, text in zip(cols, suggestions):
            if col.button(text, key=f"chip_{text}"):
                clicked = text

    log = st.container()
    typed = st.chat_input("Ask about this meeting")
    question = typed or clicked

    if question:
        history.append({"role": "user", "content": question})
        try:
            with st.spinner("Reading the transcript"):
                answer = ask_question(result["rag_chain"], question, history=history[:-1])
        except Exception as exc:  # noqa: BLE001
            answer = f"I could not get an answer: {exc}"
        history.append({"role": "assistant", "content": str(answer)})

    with log:
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def render_result() -> None:
    r = st.session_state.result

    actions = parse_items(r["action_items"])
    decisions = parse_items(r["key_decisions"])
    discussion = parse_items(r.get("discussion_points"))
    questions = parse_items(r["open_questions"])
    words = len(r["transcript"].split())
    minutes = max(1, round(words / 200))
    print(
        f"[UI] Rendering Meeting Workspace rid={r.get('rid')} "
        f"(summary_chars={len(as_text(r.get('summary')))}, actions={len(actions)}, "
        f"decisions={len(decisions)}, discussion={len(discussion)}, "
        f"questions={len(questions)}, words={words}, has_rag={r.get('rag_chain') is not None})"
    )

    language = "Hinglish, translated to English" if r["language"] == "hinglish" else "English"
    _h(
        f'<div class="brief-title">{html.escape(r["title"])}</div>'
        f'<p class="brief-meta">From {html.escape(r["source_label"])}, generated {r["generated"]}. Spoken in {language}.</p>'
        f'<p class="brief-strip">The transcript runs to about <b>{words:,} words</b>, a <b>{minutes}-minute</b> read. '
        f"It produced <b>{plural(len(actions), 'action item')}</b>, "
        f"<b>{plural(len(decisions), 'decision')}</b>, "
        f"<b>{plural(len(discussion), 'discussion point')}</b> and "
        f"<b>{plural(len(questions), 'open question')}</b>.</p>"
    )

    tabs = st.tabs([
        "Summary",
        f"Action items ({len(actions)})",
        f"Decisions ({len(decisions)})",
        f"Discussion ({len(discussion)})",
        f"Open questions ({len(questions)})",
        "Transcript",
        "Chat",
    ])

    with tabs[0]:
        st.markdown(as_text(r["summary"]))

    with tabs[1]:
        if not actions:
            st.markdown('<p class="tab-lead">No action items were found in this recording.</p>', unsafe_allow_html=True)
        else:
            fields = [action_fields(item) for item in actions]
            _h(
                '<table class="sheet"><thead><tr><th>Task</th><th>Assigned to</th><th>Deadline</th></tr></thead><tbody>'
                + "".join(
                    "<tr>"
                    f"<td>{html.escape(task)}</td><td>{html.escape(owner)}</td><td>{html.escape(deadline)}</td>"
                    "</tr>"
                    for task, owner, deadline in fields
                )
                + "</tbody></table>"
            )
            keys = [f"act_{r['rid']}_{i}" for i in range(len(fields))]
            done = sum(bool(st.session_state.get(k)) for k in keys)
            st.markdown(f'<p class="tab-lead">{done} of {len(actions)} marked done</p>', unsafe_allow_html=True)
            for key, (task, _owner, _deadline) in zip(keys, fields):
                st.checkbox(task, key=key)

    with tabs[2]:
        render_notes(r["key_decisions"], "decision", "\u2713", "No decisions were found in this recording.")

    with tabs[3]:
        render_notes(r.get("discussion_points"), "decision", "\u00b7", "No discussion points were found in this recording.")

    with tabs[4]:
        render_notes(r["open_questions"], "question", "?", "No open questions were found in this recording.")

    with tabs[5]:
        render_transcript(r["transcript"])

    with tabs[6]:
        render_chat(r)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("cached_transcript", None)
    st.session_state.setdefault("page", "Dashboard")
    inject_css()
    render_sidebar()
    page = st.session_state.page
    print(f"[UI] Rendering page={page} (has_result={st.session_state.result is not None})")
    if page == "Dashboard":
        render_dashboard()
    elif page == "Meeting Studio":
        render_landing()
    elif page == "Meeting Workspace":
        if st.session_state.result is None:
            st.info("Analyze a meeting first to unlock the workspace.")
            if st.button("Go to Meeting Studio", type="primary"):
                st.session_state.page="Meeting Studio"; st.rerun()
        else:
            render_result()
    else:
        st.session_state.page="Dashboard"; st.rerun()



main()
