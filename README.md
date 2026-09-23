# ⚡ Smart Home Energy Consumption Advisor

A hosted mini-project combining:

- **LangChain (LLM reasoning)** — extracts structured usage data (appliance load,
  daily hours, occupancy) from a free-text description, and later turns a numeric
  verdict back into a natural, personalized explanation.
- **Fuzzy Logic (scikit-fuzzy)** — a genuine Mamdani fuzzy inference system with
  membership functions, rule evaluation, and centroid defuzzification that scores
  "energy waste risk" from 0-100.
- **Streamlit UI** — a simple web app tying it together, deployable for free.

## How it works

```
User text  →  [LangChain extraction chain]  →  structured numbers
                                                       ↓
                                        [scikit-fuzzy inference engine]
                                     fuzzify → apply rules → defuzzify
                                                       ↓
                                          risk score (0-100) + band
                                                       ↓
                                  [LangChain explanation chain]  →  plain-English advice
```

- `fuzzy_engine.py` — defines 3 input variables (load, duration, occupancy) each
  with Low/Medium/High membership functions, an 11-rule rule base, and returns a
  defuzzified risk score plus the membership degrees for transparency.
- `llm_chains.py` — two LangChain pieces: `extract_usage_inputs()` (structured
  output extraction) and `explain_result()` (grounded conversational explanation).
- `app.py` — Streamlit UI that wires both together.

## File structure

```
smart-home-energy-advisor/
├── app.py                          # Streamlit UI
├── fuzzy_engine.py                 # scikit-fuzzy inference system
├── llm_chains.py                   # LangChain extraction + explanation chains
├── requirements.txt
├── .env.example                    # template for local env vars
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example        # template for Streamlit secrets
└── README.md
```

---

## 1. Run it locally

### Prerequisites
- Python 3.10+
- A free Groq API key: sign up at https://console.groq.com → API Keys → Create Key
  (Groq's free tier is used here because it's fast and free; you can swap in
  OpenAI/Gemini by editing `_get_llm()` in `llm_chains.py` if you prefer).

### Steps

```bash
# 1. Clone your repo (after you push it — see section 2)
git clone https://github.com/<your-username>/smart-home-energy-advisor.git
cd smart-home-energy-advisor

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key locally (either works)
cp .env.example .env            # then edit .env and paste your key
# OR, for Streamlit specifically:
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

# 5. Run the app
streamlit run app.py
```

It opens at `http://localhost:8501`. If you didn't set a secret, just paste your
Groq API key into the sidebar text box in the running app.

### Quick sanity check of just the fuzzy engine (no API key needed)
```bash
python3 fuzzy_engine.py
```
This prints sample risk scores for a few hardcoded scenarios so you can confirm
the fuzzy system itself works before wiring in the LLM.

---

## 2. Push to GitHub

```bash
cd smart-home-energy-advisor
git init
git add .
git commit -m "Initial commit: smart home energy advisor (LangChain + fuzzy logic + Streamlit)"

# Create an empty repo on GitHub first (github.com/new), then:
git branch -M main
git remote add origin https://github.com/<your-username>/smart-home-energy-advisor.git
git push -u origin main
```

Your `.env` and `.streamlit/secrets.toml` are excluded by `.gitignore` — **never
commit your real API key**. Only the `.example` template files are committed.

---

## 3. Host it for free — Streamlit Community Cloud (recommended)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"** → pick your `smart-home-energy-advisor` repo, branch
   `main`, main file path `app.py`.
3. Before deploying, click **"Advanced settings" → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_actual_groq_key"
   ```
4. Click **Deploy**. In a minute or two you'll get a live URL like
   `https://your-app-name.streamlit.app` — share that link.
5. Any time you `git push` new commits to `main`, the hosted app auto-updates.

### Alternative: Hugging Face Spaces
1. Go to https://huggingface.co/new-space → choose **Streamlit** as the SDK.
2. Either connect the GitHub repo or upload the same files directly (Spaces
   also reads `app.py` + `requirements.txt` at the repo root).
3. In the Space, go to **Settings → Variables and secrets** → add
   `GROQ_API_KEY` as a secret.
4. The Space builds automatically and gives you a live
   `https://huggingface.co/spaces/<you>/<space-name>` URL.

---

## Notes on the fuzzy logic (for the write-up/demo)

- **Fuzzification**: triangular membership functions (`fuzz.trimf`) map each
  crisp input (load in watts, duration in hours, occupancy count) to degrees of
  membership in Low/Medium/High sets.
- **Rule evaluation**: 11 Mamdani `IF...AND...THEN` rules combine antecedents
  using fuzzy AND (min), e.g. *"IF load is high AND duration is long THEN risk
  is high"*.
- **Defuzzification**: `skfuzzy.control.ControlSystemSimulation` aggregates all
  fired rules and applies centroid defuzzification to produce the final crisp
  0-100 risk score.
- The `/see fuzzy membership breakdown/` expander in the UI exposes the actual
  membership degrees per variable, so the fuzzy reasoning isn't a black box.

## Notes on the LangChain usage (for the write-up/demo)

- `extract_usage_inputs()` uses `with_structured_output()` bound to a Pydantic
  schema (`EnergyUsageInputs`) so the LLM must reason about vague natural
  language ("runs almost all evening", "family of four but mostly empty during
  the day") and produce sensible numeric estimates — this is real language
  understanding, not string parsing.
- `explain_result()` is a second LLM call, grounded in the fuzzy engine's actual
  numeric output, that produces a natural, non-templated explanation referencing
  the user's own scenario — not a canned string.
