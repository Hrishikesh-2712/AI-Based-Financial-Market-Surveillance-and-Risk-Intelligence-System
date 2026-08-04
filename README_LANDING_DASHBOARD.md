# Landing page + Dashboard — now one project

Your landing page (`landing.html`) and your real Bank Nifty dashboard
(`frontend/app.py` + `backend/`, `model/`, `risk_engine/`, `chatbot/`, etc.)
are now a single Streamlit app with one entrypoint:

```
app.py   <- run this (not frontend/app.py) from now on
```

## How it works

- `app.py` checks the URL query param `?view=`.
  - Nothing / `?view=home` → shows the landing page (`landing.html`), full-bleed,
    exactly as before.
  - `?view=dashboard` → shows your real live dashboard: candlestick chart,
    risk/anomaly/news panels, and the **chat panel** to ask questions
    ("Ask about anomalies & news") — that's your existing RAG chatbot from
    `chatbot/qa_chain.py`, wired up already, no changes needed there.

- Every **"Launch Dashboard"** button on the landing page links to
  `?view=dashboard` with `target="_blank"` — it opens the dashboard in a
  **new tab**. This isn't just a style choice: the landing page renders
  inside a sandboxed iframe (`components.html`, needed to keep the
  Three.js / GSAP animations working), and Streamlit's iframe sandbox does
  **not** include `allow-top-navigation`. That means `target="_top"` (same-tab
  navigation) is silently blocked by the browser on a normal click — it only
  worked before via "open link in new tab" because that bypasses the sandbox
  entirely. `allow-popups` *is* permitted, so `target="_blank"` works
  reliably on a plain click.

- Clicking **"← Home"** at the top of the dashboard clears the query param
  and takes you back to the landing page.

I updated three buttons in `landing.html`:
1. Top-right nav button ("View Dashboard" → "Launch Dashboard")
2. Hero section CTA ("View Dashboard" → "Launch Dashboard")
3. Added a new button under the "Dashboard Preview" mock section
   ("Launch Dashboard →") so people who scroll down to the fake preview
   can jump straight into the real thing.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

(Optionally also run your FastAPI backend so panels use live data instead of
the local pipeline fallback: `uvicorn backend.main:app --port 5000`.)

## Files changed / added

- `app.py` (new, project root) — the router described above.
- `frontend/app.py` — the dashboard-building code was wrapped in a
  `render_dashboard()` function so `app.py` can call it. It still works
  exactly the same if you run `streamlit run frontend/app.py` directly
  (dev/debug mode) — the `if __name__ == "__main__":` guard handles that.
- `frontend/__init__.py` (new, empty) — makes `frontend` a proper importable
  package for `from frontend.app import render_dashboard`.
- `landing.html` — the 3 button changes above. Nothing else touched (all
  animations, styling, and the in-page nav links like `#methodology`,
  `#stack`, `#team` still scroll normally).
