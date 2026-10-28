"""Streamlit chat page. Thin glue: all numbers come from the agent's answer."""

import logging
import os
import uuid

import streamlit as st

from airport_agent import config
from airport_agent.agent.graph import build_graph, make_checkpointer, run_config, turn_input
from airport_agent.agent.tools import build_tools
from airport_agent.data.faa_status import FaaStatusClient
from airport_agent.data.store import get_store
from airport_agent.llm.resilience import AllProvidersFailed
from airport_agent.llm.router import KEY_ENV, LLMRouter
from airport_agent.security.sanitize import InputRejected
from app import view

log = logging.getLogger(__name__)
st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")

HOW_SCORING_WORKS = """
**Opportunity Score (0-100)** ranks where demand is pressing on capacity, among US airports with
at least 500,000 passengers a year. Each part is a national percentile:

| Part | Weight | Measures |
|---|---|---|
| Demand | 35% | load factor: how full the planes are |
| Growth | 25% | passenger growth vs the previous 12 months |
| Terminal strain | 12.5% | taxi-in time: waiting for a free gate |
| Airside strain | 12.5% | taxi-out time and NAS delay: runway and airspace queues |
| Scale | 15% | passenger volume (log) |

**Bottleneck:** strain at or above the 66th percentile is "high". Terminal-constrained,
Airside-constrained, Both, or Unconstrained.

**Confidence** drops when an airport has little On-Time data. **Stability** checks whether the
rank holds when each weight moves ±10%.

The score ranks opportunity, not financial return.
"""


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


def init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("user_messages", 0)
    st.session_state.setdefault("pending", None)
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


def sidebar(manifest: dict) -> None:
    with st.sidebar:
        st.header("⚖️ Score weights")
        for c in config.WEIGHTS:
            st.slider(c.capitalize(), 0.0, 1.0, step=0.05, key=f"w_{c}",
                      help="Weights are renormalized to sum to 1.")
        st.button("Reset weights", on_click=reset_weights)
        st.divider()
        st.caption(f"📅 Data: {manifest['l12m_start'][:7]} to {manifest['l12m_end'][:7]} "
                   "(BTS T-100 and On-Time, latest 12 months)")
        with st.expander("❓ How scoring works"):
            st.markdown(HOW_SCORING_WORKS)
        st.button("New chat", on_click=new_chat)


def ask(text: str) -> None:
    st.session_state.pending = text


def show_answer(answer: dict, key: str) -> None:
    if answer["tldr"]:
        st.markdown(f"**{view.md_safe(answer['tldr'])}**")
    st.markdown(view.md_safe(answer["body"]))
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
    cols = st.columns(max(len(answer["follow_ups"]), 1))
    for i, (col, q) in enumerate(zip(cols, answer["follow_ups"], strict=False)):
        col.button(view.md_safe(q), key=f"{key}-fu-{i}", on_click=ask, args=(q,))


def run_turn(graph, question: str) -> None:
    if st.session_state.user_messages >= config.SESSION_MAX_MESSAGES:
        st.warning("You reached the message limit for this session. Start a new chat later.")
        return
    if not any(st.session_state[f"w_{c}"] for c in config.WEIGHTS):
        st.warning("Set at least one weight above zero.")
        return
    try:
        inp = turn_input(question, current_weights())
    except InputRejected as exc:
        st.warning(str(exc))
        return
    st.session_state.user_messages += 1
    st.session_state.history.append({"role": "user", "text": inp["question"]})
    with st.chat_message("user"):
        st.markdown(view.md_safe(inp["question"]))
    cfg = run_config(st.session_state.thread_id)
    with st.chat_message("assistant"):
        try:
            with st.status("Thinking…") as status:
                for mode, event in graph.stream(inp, cfg, stream_mode=["updates", "custom"]):
                    if mode == "custom":
                        label = view.status_label(event)
                        status.update(label=label)
                        status.write(label)
                status.update(label="Done", state="complete")
            answer = graph.get_state(cfg).values["answer"]
        except Exception as exc:
            if not isinstance(exc, AllProvidersFailed):
                log.exception("Turn failed")
            st.session_state.history.pop()  # no dangling question after a failed turn
            st.error("All AI providers are unavailable right now. Please try again in a minute."
                     if isinstance(exc, AllProvidersFailed) else
                     "Something went wrong while answering. Try rephrasing the question.")
            return
        show_answer(answer, key=f"a{len(st.session_state.history)}")
    st.session_state.history.append({"role": "assistant", "answer": answer})


def main() -> None:
    load_secrets()
    init_session()
    try:
        store, graph = resources()
    except RuntimeError as exc:  # no API key for any agent provider
        st.error(str(exc))
        st.stop()
    sidebar(store.manifest())
    st.title("✈️ Airport Investment Intelligence Agent")
    st.caption("Ranks and explains US airport renovation opportunities from public BTS and FAA "
               "data. Every number is computed in code and cited.")

    for i, turn in enumerate(st.session_state.history):
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(view.md_safe(turn["text"]))
            else:
                show_answer(turn["answer"], key=f"h{i}")

    typed = st.chat_input("Ask about US airports…", max_chars=config.MAX_INPUT_CHARS)
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if not st.session_state.history and not question:
        st.markdown("**Try one of these:**")
        for i, q in enumerate(view.EXAMPLES):
            st.button(q, key=f"ex-{i}", on_click=ask, args=(q,))

    if question:
        run_turn(graph, question)


main()
