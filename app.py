"""
AI Video Assistant - Streamlit UI
Run with:  streamlit run app.py
Place this file next to main.py (project root).
"""
import html
import os
import re
import tempfile
import time
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling (theme-agnostic: works in both Streamlit light and dark themes)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root {
  --line: rgba(128,128,128,.30);
  --panel: rgba(128,128,128,.08);
  --accent: #4f7cff;
  --warn: #d9962b;
}

html, body, .stMarkdown, button, input, textarea { font-family: 'Instrument Sans', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero-title, .result-title, .side-brand {
  font-family: 'Newsreader', Georgia, serif; font-weight: 500;
}
.hero-title { font-size: 2.6rem; line-height: 1.15; margin: 0 0 .4rem 0; }
.hero-sub { opacity: .7; font-size: 1.02rem; margin-bottom: 1.5rem; max-width: 60ch; }

.result-title { font-size: 2.2rem; line-height: 1.2; margin: 0 0 .35rem 0; overflow-wrap: anywhere; }
.result-meta { opacity: .65; font-size: .9rem; margin-bottom: 1.25rem; overflow-wrap: anywhere; }

.stat { border-left: 3px solid var(--accent); padding: .1rem 0 .1rem .8rem; }
.stat b { display: block; font-size: 1.35rem; font-weight: 600; }
.stat span { opacity: .65; font-size: .85rem; }

.st-key-summary p, .st-key-summary li {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 1.18rem; line-height: 1.7;
}
.st-key-summary { max-width: 70ch; }

.item-card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: .7rem .95rem; margin-bottom: .5rem;
  line-height: 1.5; overflow-wrap: anywhere;
}
.item-card.decision { border-left: 4px solid var(--accent); }
.item-card.question { border-left: 4px solid var(--warn); }

.transcript-box {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.3rem; max-height: 520px; overflow-y: auto;
  line-height: 1.75; font-size: .98rem; overflow-wrap: anywhere;
}
.transcript-box mark { background: rgba(255,200,0,.45); color: inherit; padding: 0 2px; border-radius: 3px; }

.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
.stTabs [data-baseweb="tab"] { padding: .6rem 0; font-weight: 500; }

/* full-width buttons without relying on use_container_width */
section[data-testid="stSidebar"] .stButton > button,
[data-testid="stColumn"] .stButton > button,
[data-testid="column"] .stButton > button,
.stDownloadButton > button { width: 100%; }
.stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }

.side-brand { font-size: 1.5rem; }
.side-note { opacity: .65; font-size: .85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
st.session_state.setdefault("history", [])   # list of result dicts
st.session_state.setdefault("active", None)  # index into history

SUGGESTIONS = [
    "What were the main topics discussed?",
    "Who is responsible for what?",
    "What deadlines were mentioned?",
    "Give me a 3-bullet recap.",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def esc(text) -> str:
    """Escape text for raw HTML inside st.markdown (also blocks $ LaTeX parsing)."""
    return html.escape(str(text)).replace("$", "&#36;")


def md(text) -> str:
    """Escape text for markdown so '$' is never treated as LaTeX."""
    return str(text).replace("$", r"\$")


def to_items(value) -> list[str]:
    """Normalise LLM output (string or list) into a clean list of items."""
    if value is None:
        return []
    raw = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).splitlines()
    items = []
    for line in raw:
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def save_upload(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def highlight(text: str, query: str):
    """Return (safe single-line HTML, match count). Matches the raw text, so
    searching for things like 'amp' can never break HTML entities."""
    query = query.strip()
    if not query:
        return esc(text).replace("\n", "<br>"), 0
    parts = re.split(f"({re.escape(query)})", text, flags=re.IGNORECASE)
    out, count = [], 0
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(f"<mark>{esc(part)}</mark>")
            count += 1
        else:
            out.append(esc(part))
    return "".join(out).replace("\n", "<br>"), count


def build_markdown_report(r: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) or "_None found_"

    return (
        f"# {r['title']}\n\n"
        f"_Generated {r['created']} from {r['source_label']}_\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action items\n{bullets(to_items(r['action_items']))}\n\n"
        f"## Key decisions\n{bullets(to_items(r['key_decisions']))}\n\n"
        f"## Open questions\n{bullets(to_items(r['open_questions']))}\n\n"
        f"## Transcript\n{r['transcript']}\n"
    )


def run_with_progress(source: str, source_label: str, language: str):
    """Run the pipeline step by step, showing live progress."""
    started = time.time()
    result = {
        "id": uuid.uuid4().hex[:8],
        "source_label": source_label,
        "language": language,
    }

    with st.status("Processing your video…", expanded=True) as status:
        try:
            st.write("Fetching and splitting audio")
            chunks = process_input(source)
            result["chunks"] = len(chunks)

            st.write(f"Transcribing {len(chunks)} audio chunk(s)")
            transcript = transcribe_all(chunks, language)
            if not transcript or not str(transcript).strip():
                raise ValueError("The transcription came back empty.")
            result["transcript"] = str(transcript)

            st.write("Writing title and summary")
            result["title"] = str(generate_title(transcript)).strip().strip('"') or "Untitled video"
            result["summary"] = str(summarize(transcript))

            st.write("Extracting action items, decisions and questions")
            result["action_items"] = extract_action_items(transcript)
            result["key_decisions"] = extract_key_decisions(transcript)
            result["open_questions"] = extract_questions(transcript)

            st.write("Indexing the transcript for chat")
            result["rag_chain"] = build_rag_chain(transcript)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(
                f"**Something went wrong:** {exc}\n\n"
                "Check that the link or file is valid, that your API keys are set "
                "in `.env`, and try again."
            )
            return None

        result["elapsed"] = time.time() - started
        result["created"] = datetime.now().strftime("%d %b %Y, %H:%M")
        result["chat"] = []
        status.update(label="Done", state="complete", expanded=False)
    return result


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">🎬 Video Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note">Turn any video or meeting into notes you can question.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    input_mode = st.radio("Source", ["YouTube link", "Upload file"], horizontal=True)

    source, source_label = None, None
    if input_mode == "YouTube link":
        url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=…")
        if url.strip():
            source, source_label = url.strip(), url.strip()
    else:
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded:
            source, source_label = uploaded, uploaded.name

    language = st.selectbox("Spoken language", ["english", "hinglish"])

    go = st.button("Analyze video", type="primary", disabled=source is None)

    if st.session_state.history:
        st.divider()
        st.markdown("**Recent**")
        for idx in reversed(range(len(st.session_state.history))):
            item = st.session_state.history[idx]
            label = item["title"][:40] + ("…" if len(item["title"]) > 40 else "")
            is_active = st.session_state.active == idx
            if st.button(md(label), key=f"hist_{item['id']}",
                         type="primary" if is_active else "secondary"):
                st.session_state.active = idx
                st.rerun()
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.active = None
            st.rerun()

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if go and source is not None:
    is_upload = input_mode == "Upload file"
    path = save_upload(source) if is_upload else source
    try:
        res = run_with_progress(path, source_label, language)
    finally:
        if is_upload and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    if res:
        st.session_state.history.append(res)
        st.session_state.active = len(st.session_state.history) - 1
        st.rerun()

# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------
if st.session_state.active is None or not st.session_state.history:
    st.markdown('<div class="hero-title">Paste a video. Get the meeting notes.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Add a YouTube link or upload a recording in the sidebar. '
        "You'll get a transcript, summary, action items, decisions and open questions, "
        "and you can ask follow-up questions about anything said.</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Add a source**  \nYouTube link or a local audio/video file.")
    c2.markdown("**2. Analyze**  \nTranscription and extraction run in one go.")
    c3.markdown("**3. Ask questions**  \nChat answers come from the transcript.")
    st.stop()

# --------------------------------------------------------------------------
# Result view
# --------------------------------------------------------------------------
r = st.session_state.history[st.session_state.active]
rid = r["id"]
words = len(r["transcript"].split())
read_min = max(1, round(words / 220))
actions = to_items(r["action_items"])
decisions = to_items(r["key_decisions"])
questions = to_items(r["open_questions"])

st.markdown(f'<div class="result-title">{esc(r["title"])}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="result-meta">{esc(r["source_label"])} &nbsp;|&nbsp; '
    f'{r["created"]} &nbsp;|&nbsp; {esc(r["language"].capitalize())}</div>',
    unsafe_allow_html=True,
)

stats = [
    (f"{words:,}", "words transcribed"),
    (f"{read_min} min", "to read"),
    (str(len(actions)), "action items"),
    (f"{r['elapsed']:.0f}s", "processing time"),
]
for col, (num, label) in zip(st.columns(4), stats):
    col.markdown(f'<div class="stat"><b>{num}</b><span>{label}</span></div>', unsafe_allow_html=True)

st.write("")

tab_overview, tab_items, tab_chat, tab_transcript, tab_export = st.tabs(
    ["Summary", "Action items & decisions", "Chat", "Transcript", "Export"]
)

# ---- Summary ----
with tab_overview:
    with st.container(key="summary"):
        st.markdown(md(r["summary"]))

# ---- Items ----
with tab_items:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Action items")
        if not actions:
            st.info("No action items were found in this video.")
        for i, item in enumerate(actions):
            st.checkbox(md(item), key=f"act_{rid}_{i}")

    with right:
        st.subheader("Key decisions")
        if not decisions:
            st.info("No decisions were found in this video.")
        for item in decisions:
            st.markdown(f'<div class="item-card decision">{esc(item)}</div>', unsafe_allow_html=True)

        st.subheader("Open questions")
        if not questions:
            st.info("No open questions were found in this video.")
        for item in questions:
            st.markdown(f'<div class="item-card question">{esc(item)}</div>', unsafe_allow_html=True)

# ---- Chat ----
with tab_chat:
    chat = r["chat"]
    pending = st.session_state.pop("pending_q", None)

    if not chat and not pending:
        st.caption("Try one of these, or type your own question below.")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 2].button(s, key=f"sugg_{rid}_{i}"):
                st.session_state["pending_q"] = s
                st.rerun()

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(md(msg["content"]))

    typed = st.chat_input("Ask about this video…", key=f"chat_input_{rid}")
    question = typed or pending

    if question:
        chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript…"):
                try:
                    answer = str(ask_question(r["rag_chain"], question))
                except Exception as exc:  # noqa: BLE001
                    answer = f"I couldn't answer that: {exc}"
            st.markdown(md(answer))
        chat.append({"role": "assistant", "content": answer})

    if chat and st.button("Clear chat", key=f"clear_chat_{rid}"):
        r["chat"] = []
        st.rerun()

# ---- Transcript ----
with tab_transcript:
    query = st.text_input("Search the transcript", placeholder="Type a word or phrase", key=f"tsearch_{rid}")
    body, matches = highlight(r["transcript"], query)
    if query.strip():
        st.caption(f"{matches} match(es)" if matches else "No matches found.")
    st.markdown(f'<div class="transcript-box">{body}</div>', unsafe_allow_html=True)

# ---- Export ----
with tab_export:
    st.write("Download everything from this session.")
    slug = re.sub(r"[^a-z0-9]+", "-", r["title"].lower()).strip("-")[:50] or "video-notes"
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download report (.md)",
        data=build_markdown_report(r),
        file_name=f"{slug}.md",
        mime="text/markdown",
    )
    c2.download_button(
        "Download transcript (.txt)",
        data=r["transcript"],
        file_name=f"{slug}-transcript.txt",
        mime="text/plain",
    )
