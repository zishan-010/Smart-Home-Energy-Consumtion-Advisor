"""
llm_chains.py
--------------
Two LangChain pieces that do real reasoning/language work:

1. ExtractionChain  — reads a free-text description of a household's energy usage
   and pulls out structured numeric fields (appliance load in watts, daily usage
   hours, occupancy) using an LLM with structured output. This is NOT regex/keyword
   matching — the model has to interpret vague, inconsistent natural language
   ("runs almost all evening", "family of four but mostly empty during the day")
   and estimate sensible numbers.

2. ExplanationChain — takes the fuzzy engine's numeric verdict (score + band +
   membership degrees) and turns it back into a natural, personalized, conversational
   recommendation for the user, referencing their original query.

Uses Groq's free-tier LLM API via langchain-groq (fast + free for small projects).
Swap `ChatGroq` for `ChatOpenAI` / `ChatGoogleGenerativeAI` if you prefer another
provider — the rest of the chain logic is provider-agnostic.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ----------------------------------------------------------------------
# 1. Structured extraction schema
# ----------------------------------------------------------------------
class EnergyUsageInputs(BaseModel):
    """Structured fields extracted from a user's free-text energy usage description."""

    appliance_load_watts: float = Field(
        description=(
            "Estimated total power draw in watts of the appliance(s) being discussed. "
            "If the user names an appliance without wattage, estimate a realistic typical "
            "value (e.g. window AC ~1200-1500W, split AC ~1500-2000W, central AC ~3000-4500W, "
            "refrigerator ~150-400W, water heater ~2000-4500W, washing machine ~500-1000W). "
            "If multiple appliances are mentioned, sum a reasonable combined load."
        )
    )
    daily_usage_hours: float = Field(
        description=(
            "Estimated number of hours per day the appliance(s) run, inferred from phrases "
            "like 'all evening' (~5), 'all day' (~14-16), 'overnight' (~8), 'a couple hours' (~2)."
        )
    )
    occupancy_count: float = Field(
        description=(
            "Estimated number of people typically present in the home while the appliance runs. "
            "If the text implies the house is often empty during usage, use a low number even if "
            "total household size is larger."
        )
    )
    appliance_summary: str = Field(
        description="A short (<=8 word) label for what appliance/usage this is, e.g. 'split AC, evenings'."
    )


def _get_llm(api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b", temperature: float = 0.2):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No Groq API key found. Set GROQ_API_KEY as an environment variable / Streamlit secret, "
            "or pass it in from the sidebar."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=key)


def extract_usage_inputs(user_query: str, api_key: Optional[str] = None) -> EnergyUsageInputs:
    """
    LLM-powered extraction: natural language -> structured numeric inputs for the
    fuzzy engine. This is genuine language understanding, not keyword matching.
    """
    llm = _get_llm(api_key)
    structured_llm = llm.with_structured_output(EnergyUsageInputs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an energy-auditing assistant. Read the user's description of how they use "
                "a home appliance and estimate the three numeric fields needed for an energy analysis. "
                "Make sensible real-world assumptions when the user is vague, and never leave a field "
                "empty. Think about what's actually reasonable for an Indian or generic urban household "
                "unless told otherwise.",
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | structured_llm
    result: EnergyUsageInputs = chain.invoke({"query": user_query})
    return result


# ----------------------------------------------------------------------
# 2. Conversational explanation chain
# ----------------------------------------------------------------------
_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly, practical home energy advisor. You are given the user's original "
            "question, the numeric inputs a system inferred from it, and the output of a fuzzy logic "
            "risk engine (a 0-100 energy-waste risk score with a Low/Medium/High band, plus how strongly "
            "the inputs belonged to each fuzzy category). Explain the verdict in plain, conversational "
            "language: (1) briefly restate what you understood about their usage, (2) explain WHY the "
            "score came out the way it did in terms a non-technical person understands (avoid the words "
            "'fuzzy', 'membership', or 'defuzzification' — just talk about high/moderate/low load, duration, "
            "occupancy in plain terms), and (3) give 2-4 concrete, specific action items to reduce waste if "
            "risk is Medium/High, or a brief affirmation plus one optimization tip if Low. Keep it under "
            "180 words. Do not use markdown headers.",
        ),
        (
            "human",
            "User's original question: {query}\n\n"
            "Inferred inputs: appliance load ~{load_w:.0f}W, usage ~{duration_hr:.1f} hrs/day, "
            "~{occupancy:.0f} people typically present.\n\n"
            "Fuzzy engine verdict: risk score {score}/100 ({band} risk).\n"
            "Category strengths -> load: {load_mem}; duration: {duration_mem}; occupancy: {occupancy_mem}.\n\n"
            "Write the explanation now.",
        ),
    ]
)


def explain_result(user_query: str, extracted: EnergyUsageInputs, fuzzy_result: dict, api_key: Optional[str] = None) -> str:
    """
    LLM-powered explanation: turns numeric fuzzy engine output into a natural,
    personalized recommendation grounded in the user's original words.
    """
    llm = _get_llm(api_key, temperature=0.4)
    chain = _EXPLANATION_PROMPT | llm | StrOutputParser()

    def top_label(mem: dict) -> str:
        return max(mem, key=mem.get)

    m = fuzzy_result["memberships"]

    response = chain.invoke(
        {
            "query": user_query,
            "load_w": fuzzy_result["inputs"]["load_w"],
            "duration_hr": fuzzy_result["inputs"]["duration_hr"],
            "occupancy": fuzzy_result["inputs"]["occupancy_n"],
            "score": fuzzy_result["score"],
            "band": fuzzy_result["band"],
            "load_mem": top_label(m["load"]),
            "duration_mem": top_label(m["duration"]),
            "occupancy_mem": top_label(m["occupancy"]),
        }
    )
    return response
