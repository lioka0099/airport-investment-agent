"""Streamlit chat page. Thin glue: all numbers come from the agent's answer."""

import logging
import os
import uuid

import streamlit as st
from google import genai
from openai import OpenAI

from airport_agent import config
from airport_agent.agent.graph import build_graph, make_checkpointer, run_config, turn_input
from airport_agent.agent.tools import build_tools
from airport_agent.data.faa_status import FaaStatusClient
from airport_agent.data.store import get_store
from airport_agent.llm.resilience import AllProvidersFailed
from airport_agent.llm.router import KEY_ENV, LLMRouter
from airport_agent.security.sanitize import InputRejected
from app import view, voice

log = logging.getLogger(__name__)
st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")

# Shown in each slider's "?" tooltip; the defaults come from config.WEIGHTS.
WEIGHT_HELP = {
    "demand": "Load factor: how full the planes are.",
    "growth": "Passenger growth vs the previous 12 months.",
    "terminal": "Taxi-in time: waiting for a free gate. At or above the 66th percentile, the "
                "airport counts as terminal-constrained.",
    "airside": "Taxi-out time and NAS delay: runway and airspace queues. At or above the 66th "
               "percentile, the airport counts as airside-constrained.",
    "scale": "Passenger volume (log scale).",
}
SCORE_NOTE = ("National percentiles among US airports with 500k+ passengers. Weights are "
              "renormalized to sum to 1. Ranks opportunity, not financial return.")

HEADER_CSS = """<style>
/* Dusk sky: the title carries the violet-to-fuchsia gradient; amber is the one warm accent. */
.hdr { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
.hdr h1 { margin: 0; padding: 0; font: 700 2rem "Sora", sans-serif; letter-spacing: -.02em;
          background: linear-gradient(90deg, #4C1D95, #7C3AED 50%, #C026D3);
          -webkit-background-clip: text; background-clip: text; color: transparent; }
.sign { font: 700 .8rem "Sora", sans-serif; letter-spacing: .12em; color: #6D28D9;
        background: #F1ECFD; border: 1px solid #E0D6F8; padding: .2rem .65rem;
        border-radius: 999px; }
.sign::before { content: "●"; color: #F59E0B; margin-right: .4rem; }
.sub { margin: .5rem 0 .25rem; color: #4A3D66; }
.meta { margin: 0; font-family: "JetBrains Mono", monospace; font-size: .76rem; color: #7A6E94; }
/* Suggested questions and buttons read as pills. */
.stButton button { border-radius: 999px; }
.stButton button[kind="secondary"]:hover { border-color: #7C3AED; color: #6D28D9;
                                          background: #F1ECFD; }
[data-testid="stChatInput"] > div { border-radius: 999px; }
/* Bubbles: the user's in deeper lavender, the agent's in a pale one. The user keeps Streamlit's
   default avatar because only that one carries a role test id to tell the two apart. */
[data-testid="stChatMessage"] { padding: 1rem 1.1rem; border-radius: 18px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #EADFFD; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
  background: #F7F3FF; border: 1px solid #E6DCFA; }
[data-testid="stChatMessageAvatarUser"] { background: #F59E0B; color: #FFFFFF; }
[data-testid="stChatMessageAvatarCustom"] { background: #7C3AED; color: #FFFFFF; }
/* The default toggle is a bare chevron; a label says what is behind it. Streamlit's test id is
   not a public API, so a version bump may drop this; the toggle still works without it. */
[data-testid="stExpandSidebarButton"] { width: auto; padding: 0 .6rem; gap: .35rem;
  border: 1px solid #E0D6F8; border-radius: 999px; background: #FFFFFF; }
[data-testid="stExpandSidebarButton"]::after { content: "Score weights"; font-size: .85rem;
  font-weight: 600; color: #6D28D9; }
/* Top bar in violet, with New chat pinned on its right (z-index above Streamlit's bar). */
[data-testid="stHeader"] { background: linear-gradient(90deg, #6D28D9, #8B5CF6 60%, #A855F7); }
[data-testid="stHeader"] [data-testid="stToolbar"] button { color: #FFFFFF; }
.st-key-new_chat { position: fixed; top: .55rem; right: 1rem; z-index: 999991; width: auto; }
.st-key-new_chat button { background: #FFFFFF; border-color: #FFFFFF; color: #6D28D9;
                          font-weight: 600; }
.st-key-new_chat button:hover { background: #F1ECFD; color: #6D28D9; }
[data-testid="stSidebarUserContent"] { padding-bottom: 1rem; }  /* weights fit without scrolling */
[data-testid="stAudio"] { display: none; }  /* the agent talks; it doesn't hand over a file */
</style>"""

# Streamlit can't see the audio end, the browser can: hide "Stop voice" then. No LLM text in here.
HIDE_STOP_JS = """<script>(function wait() {
  const a = document.querySelector("audio"), stop = document.querySelector(".st-key-STOP_KEY");
  if (!a || !stop) return setTimeout(wait, 100);
  a.addEventListener("ended", () => stop.style.display = "none");
})();</script>"""

AVATAR = {"user": None, "assistant": ":material/flight_takeoff:"}

PROVIDERS_DOWN = "All AI providers are unavailable right now. Please try again in a minute."
TURN_FAILED = "Something went wrong while answering. Try again or rephrase the question."
LIMIT_REACHED = "You reached the message limit for this session. Start a new chat later."
COULD_NOT_HEAR = "Couldn't hear that — try again or type."


def load_secrets() -> None:
    try:
        for name in [*KEY_ENV.values(), "LANGSMITH_API_KEY", "LANGSMITH_TRACING",
                     "LANGSMITH_PROJECT"]:
            if name in st.secrets:
                os.environ.setdefault(name, str(st.secrets[name]))
    except Exception:  # no secrets.toml locally: environment variables are used instead
        pass


@st.cache_resource
def resources():
    store = get_store()
    tools = build_tools(store, FaaStatusClient())
    graph = build_graph(LLMRouter(), tools, make_checkpointer(), store.manifest())
    return store, graph


@st.cache_resource
def voice_clients() -> dict:
    """A client per voice provider that has a key, so the next one can take over."""
    make = {"openai": lambda: OpenAI(timeout=20, max_retries=0),
            "google": lambda: genai.Client(http_options={"timeout": 20_000})}
    return {p: f() for p, f in make.items() if os.environ.get(KEY_ENV[p])}


def init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("user_messages", 0)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("failed", None)
    for c, w in config.WEIGHTS.items():
        st.session_state.setdefault(f"w_{c}", w)


def current_weights() -> dict | None:
    w = {c: st.session_state[f"w_{c}"] for c in config.WEIGHTS}
    return None if w == config.WEIGHTS else w


def reset_weights() -> None:
    for c, w in config.WEIGHTS.items():
        st.session_state[f"w_{c}"] = w


def new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.failed = None


def sidebar(busy: bool) -> None:
    with st.sidebar:
        st.subheader("⚖️ Score weights")
        st.caption(SCORE_NOTE)
        for c, default in config.WEIGHTS.items():
            st.slider(c.capitalize(), 0.0, 1.0, step=0.025, key=f"w_{c}",
                      help=f"{WEIGHT_HELP[c]} Default {default * 100:g}%.", disabled=busy)
        st.button("Reset weights", on_click=reset_weights, disabled=busy)


def header(manifest: dict, busy: bool) -> None:
    """Title in a dusk gradient, and a status line that shows the data period and
    whether custom weights are on, so neither hides in the collapsed sidebar."""
    weights = "custom weights (sidebar)" if current_weights() else "default weights"
    # CSS pins it to the top bar, so it stays in reach while the chat scrolls
    st.button("New chat", icon=":material/add_comment:", on_click=new_chat, width="content",
              disabled=busy, key="new_chat")
    st.html(f"""{HEADER_CSS}
<div class="hdr"><span class="sign">IAI</span><h1>Airport Investment Intelligence</h1></div>
<p class="sub">Ranks and explains US airport renovation opportunities from public BTS and FAA
data. Every number is computed in code and cited.</p>
<p class="meta">BTS T-100 and On-Time, {manifest['l12m_start'][:7]} to
 {manifest['l12m_end'][:7]} · {weights}</p>""")


def bubble(role: str):
    return st.chat_message(role, avatar=AVATAR[role])


def ask(text: str) -> None:
    st.session_state.pending = text


def written(answer: dict) -> None:
    if answer["tldr"]:
        st.markdown(f"**{view.md_safe(answer['tldr'])}**")
    st.markdown(view.md_safe(answer["body"]))


def show_answer(answer: dict, key: str, follow_ups: bool = False, speech=None) -> None:
    if speech:  # drawn with the text, so the two start together
        audio, mime = speech
        st.audio(audio, format=mime, autoplay=True)
        # Any rerun drops the audio, so this stops it, and so does asking something new.
        st.button("Stop voice", icon=":material/stop_circle:", type="tertiary", key=f"stop-{key}")
        st.html(HIDE_STOP_JS.replace("STOP_KEY", f"stop-{key}"), unsafe_allow_javascript=True)
    if spoken := answer.get("spoken"):  # voice turn: show what was heard, the rest on demand
        st.markdown(view.md_safe(spoken))
        with st.expander("Full answer"):
            written(answer)
    else:
        written(answer)
    badges = [view.verification_badge(answer["verification"])]
    if conf := view.confidence(answer["tool_data"]):
        badges.append(f"Confidence: {conf}")
    st.caption(" · ".join(badges))
    if chart := view.chart(answer["tool_data"]):
        st.altair_chart(chart)
    if answer["citations"]:
        with st.expander("Sources"):
            st.markdown(view.sources_md(answer["citations"]))
    if tables := view.tables(answer["tool_data"]):
        with st.expander("Data"):
            for i, (title, df) in enumerate(tables):
                st.markdown(f"**{title}**")
                st.dataframe(df, hide_index=True)
                st.download_button("Download CSV", df.to_csv(index=False),
                                   file_name=f"{title.replace(' ', '_').lower()}.csv",
                                   key=f"{key}-csv-{i}")
    if answer["tool_data"]:
        with st.expander("How I got this"):
            st.markdown(view.steps_md(answer["tool_data"]))
            st.caption(f"Written by {answer['provider']}. Every number is filled in by code "
                       "from tool results, not typed by the AI.")
    with st.expander("Assumptions & coverage"):
        st.markdown(f"Data period: {answer['period']}")
        for a in answer["assumptions"]:
            st.markdown(f"- {view.md_safe(a)}")
    if notes := view.reviewer_notes(answer["verification"]):
        with st.expander("Reviewer notes"):
            st.caption("Suggestions from the second AI. They did not block verification.")
            for n in notes:
                st.markdown(f"- {n}")
    slot = st.empty()  # a placeholder replaces stale buttons at once, so none linger faded
    if follow_ups and answer["follow_ups"]:
        with slot.container():
            st.caption("Ask next")  # sets suggestions apart from the answer above
            for i, q in enumerate(answer["follow_ups"]):
                st.button(view.md_safe(q), key=f"{key}-fu-{i}", on_click=ask, args=(q,),
                          wrap=True)


def notify(error: str, retry: str | None = None) -> None:
    """Show `error` after the rerun; kept in state so it survives reruns until the next turn."""
    st.session_state.failed = {"error": error, "question": retry}
    st.rerun()


def heard(audio) -> str | None:
    if st.session_state.user_messages >= config.SESSION_MAX_MESSAGES:
        return None  # no paid transcription past the limit; run_turn is never reached
    try:
        return voice.transcribe(audio.getvalue(), voice_clients()) or None  # silence -> ""
    except Exception as exc:
        log.warning("STT failed: %s", type(exc).__name__)  # type only: may hold key
        return None


def prepare_speech(answer: dict) -> None:
    """Asked by voice -> answered by voice, in words written for speech."""
    try:
        said = answer.get("spoken") or answer["tldr"] or answer["body"]
        st.session_state.speak = voice.speak(said, voice_clients())
    except Exception as exc:  # the written answer still shows
        log.warning("TTS failed: %s", type(exc).__name__)


def run_turn(graph, question: str, by_voice: bool = False) -> None:
    """Answer one question. Always ends in a rerun so the page redraws idle."""
    if st.session_state.user_messages >= config.SESSION_MAX_MESSAGES:
        notify(LIMIT_REACHED)
    if not any(st.session_state[f"w_{c}"] for c in config.WEIGHTS):
        notify("Set at least one weight above zero.")
    try:
        inp = turn_input(question, current_weights(), by_voice)
    except InputRejected as exc:
        notify(str(exc))
    st.session_state.user_messages += 1
    st.session_state.failed = None
    st.session_state.history.append({"role": "user", "text": inp["question"]})
    with bubble("user"):
        st.markdown(view.md_safe(inp["question"]))
    cfg = run_config(st.session_state.thread_id)
    with bubble("assistant"):
        try:
            with st.status("Thinking…") as status:
                for mode, event in graph.stream(inp, cfg, stream_mode=["updates", "custom"]):
                    if mode == "custom":
                        label = view.status_label(event)
                        status.update(label=label)
                        status.write(label)
                answer = graph.get_state(cfg).values["answer"]
                if by_voice:  # voice first, so the text and the speech start together
                    status.update(label="Preparing voice…")
                    prepare_speech(answer)
                status.update(label="Done", state="complete")
        except Exception as exc:
            if not isinstance(exc, AllProvidersFailed):
                log.exception("Turn failed")
            st.session_state.history.pop()  # no dangling question after a failed turn
            notify(PROVIDERS_DOWN if isinstance(exc, AllProvidersFailed) else TURN_FAILED,
                   retry=question)
        if not by_voice:  # a voice answer is drawn after the rerun, with its subtitles
            show_answer(answer, key=f"a{len(st.session_state.history)}")
    st.session_state.history.append({"role": "assistant", "answer": answer})
    st.rerun()


def main() -> None:
    load_secrets()
    init_session()
    try:
        store, graph = resources()
    except RuntimeError as exc:  # no API key for any agent provider
        st.error(str(exc))
        st.stop()
    pending, st.session_state.pending = st.session_state.pending, None
    value = st.chat_input("Ask about US airports, or tap the mic…", key="chat",
                          max_chars=config.MAX_INPUT_CHARS,
                          accept_audio=bool(voice_clients()),
                          submit_mode="disable", disabled=pending is not None)
    question, by_voice = pending, False
    if value is not None and getattr(value, "audio", None) is not None:
        question, by_voice = heard(value.audio), True
        if question is None:
            over = st.session_state.user_messages >= config.SESSION_MAX_MESSAGES
            st.session_state.failed = {"error": LIMIT_REACHED if over else COULD_NOT_HEAR,
                                       "question": None}
    elif value is not None:
        question = getattr(value, "text", value) or None

    # Drawn after the question is known: any widget change reruns the script and would cut off
    # the answer being written, so the controls are locked while a turn runs.
    sidebar(busy=bool(question))
    header(store.manifest(), busy=bool(question))

    last = len(st.session_state.history) - 1
    for i, turn in enumerate(st.session_state.history):
        with bubble(turn["role"]):
            if turn["role"] == "user":
                st.markdown(view.md_safe(turn["text"]))
            else:
                speech = st.session_state.pop("speak", None) if i == last else None
                show_answer(turn["answer"], key=f"h{i}", follow_ups=i == last and not question,
                            speech=speech)

    idle = st.empty()  # chips and errors: cleared at once while a turn runs
    if not question:
        with idle.container():
            if failed := st.session_state.failed:
                with bubble("assistant"):
                    st.error(failed["error"])
                    if failed["question"]:
                        st.button("Try again", key="retry", on_click=ask,
                                  args=(failed["question"],))
            if not st.session_state.history:
                st.markdown("**Example questions**")
                for i, q in enumerate(view.EXAMPLES):
                    st.button(q, key=f"ex-{i}", on_click=ask, args=(q,))

    if question:
        run_turn(graph, question, by_voice)

main()
