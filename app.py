from threading import Lock
from time import perf_counter

import streamlit as st

from main import build_scheduler


st.set_page_config(page_title="PagedServe", page_icon="⚙️")


@st.cache_resource
def get_runtime():
    return build_scheduler(), Lock()


scheduler, scheduler_lock = get_runtime()

st.title("PagedServe")
st.caption("Continuous-batching LLM inference with a paged KV cache")
st.info(
    "The included reference model uses locally initialized weights to exercise "
    "the complete serving pipeline; its generated text is not semantically trained output."
)

with st.form("generation-form"):
    prompt = st.text_area("Prompt", value="Orca and paged attention")
    max_new_tokens = st.slider("Maximum new tokens", 1, 64, 12)
    submitted = st.form_submit_button("Generate")

if submitted:
    started_at = perf_counter()

    try:
        with st.spinner("Running inference..."):
            with scheduler_lock:
                request_id = scheduler.add_request(
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
                while request_id not in scheduler.finished:
                    scheduler.step()

                request_state = scheduler.finished[request_id]
                generated_text = scheduler.tokenizer.decode(
                    request_state.generated_token_ids
                )

        latency_ms = (perf_counter() - started_at) * 1000
        request_column, latency_column, token_column = st.columns(3)
        request_column.metric("Request", request_id)
        latency_column.metric("Latency", f"{latency_ms:.2f} ms")
        token_column.metric("Tokens", len(request_state.generated_token_ids))

        st.subheader("Generated text")
        st.code(repr(generated_text), language=None)

        st.subheader("Generated token IDs")
        st.code(str(request_state.generated_token_ids), language=None)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
