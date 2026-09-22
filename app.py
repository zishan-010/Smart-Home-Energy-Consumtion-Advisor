"""
app.py
-------
Streamlit front-end for the Smart Home Energy Consumption Advisor.

Flow:
  1. User describes their appliance usage in plain English.
  2. LangChain extraction chain turns that into structured numbers.
  3. The scikit-fuzzy inference system scores energy-waste risk (0-100).
  4. LangChain explanation chain turns the verdict back into friendly advice.
"""

import streamlit as st

from fuzzy_engine import compute_risk
from llm_chains import extract_usage_inputs, explain_result

st.set_page_config(page_title="Smart Home Energy Advisor", page_icon="⚡", layout="centered")

st.title("⚡ Smart Home Energy Consumption Advisor")
st.caption(
    "Describe how you use an appliance in plain English. An LLM (via LangChain) extracts the "
    "numbers, a fuzzy logic engine scores your energy-waste risk, and the LLM explains the result."
)

# ---------------------------------------------------------------------
# Sidebar: API key handling
#
# If a GROQ_API_KEY secret is already configured (locally via
# .streamlit/secrets.toml, or on Streamlit Community Cloud via the app's
# Secrets settings), we use it directly and NEVER put it into a visible
# widget — a text_input's value can always be revealed with the eye icon,
# which would leak your key to anyone using your hosted app. We only show
# an input box when no secret is set, so each visitor supplies their own key.
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    try:
        secret_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError, Exception):
        secret_key = ""

    if secret_key:
        api_key = secret_key
        st.success("Groq API key loaded from secrets.", icon="🔒")
    else:
        api_key = st.text_input(
            "Groq API key",
            value="",
            type="password",
            help="Get a free key at console.groq.com. Your key is only used for this session and never displayed.",
        )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. LangChain extracts load (W), usage hours/day, and occupancy from your text.\n"
        "2. A Mamdani fuzzy inference system (membership functions → rules → centroid "
        "defuzzification) computes an energy-waste risk score.\n"
        "3. LangChain writes a plain-English explanation grounded in that score."
    )

# ---------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------
example = "My split AC runs almost the whole evening after work, usually just me and my partner home, set around 22°C."
user_query = st.text_area(
    "Describe your appliance usage",
    placeholder=example,
    height=100,
)

analyze = st.button("Analyze my energy usage", type="primary")

if analyze:
    if not user_query.strip():
        st.warning("Please describe your appliance usage first.")
        st.stop()
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar (free at console.groq.com).")
        st.stop()

    with st.spinner("Reading your description..."):
        try:
            extracted = extract_usage_inputs(user_query, api_key=api_key)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    with st.spinner("Running fuzzy inference..."):
        fuzzy_result = compute_risk(
            load_w=extracted.appliance_load_watts,
            duration_hr=extracted.daily_usage_hours,
            occupancy_n=extracted.occupancy_count,
        )

    with st.spinner("Writing your personalized explanation..."):
        try:
            explanation = explain_result(user_query, extracted, fuzzy_result, api_key=api_key)
        except Exception as e:
            st.error(f"Explanation generation failed: {e}")
            st.stop()

    # ---- Results ----
    st.subheader(f"{extracted.appliance_summary}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated load", f"{extracted.appliance_load_watts:.0f} W")
    col2.metric("Usage / day", f"{extracted.daily_usage_hours:.1f} hrs")
    col3.metric("Typical occupancy", f"{extracted.occupancy_count:.0f} people")

    band_color = {"Low": "green", "Medium": "orange", "High": "red"}[fuzzy_result["band"]]
    st.markdown(
        f"### Energy Waste Risk: :{band_color}[{fuzzy_result['band']}]  ({fuzzy_result['score']}/100)"
    )
    st.progress(min(int(fuzzy_result["score"]), 100) / 100)

    with st.expander("See fuzzy membership breakdown (how the score was derived)"):
        m = fuzzy_result["memberships"]
        for var_name, degrees in m.items():
            st.write(f"**{var_name}**: " + ", ".join(f"{k}={v:.2f}" for k, v in degrees.items()))

    st.markdown("### Advisor's take")
    st.write(explanation)

st.markdown("---")
st.caption("Built with LangChain + scikit-fuzzy + Streamlit.")