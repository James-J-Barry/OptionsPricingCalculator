# optvol — Volatility Surface & Model-Divergence Engine

A Python engine that pulls **live option chains**, **solves** the implied
volatility of every contract by inverting Black-Scholes against the market mid
price, and surfaces where a naive single-volatility model **diverges from the
market** — i.e. it recovers the volatility smile/skew that the market is pricing
in.

> **The core idea.** Implied volatility is *defined* as the volatility that makes
> the Black-Scholes price equal the market price. A common beginner mistake is to
> feed the market's quoted IV *into* Black-Scholes — which is circular and
> surfaces no new information. This project does the opposite: it **inverts** the
> relationship to solve for IV from observed prices, then quantifies the
> information a flat-vol model throws away.

## What it does (Phase 1)

- **Live data** — pulls option chains, spot, dividend yield, and a real
  risk-free rate (13-week T-bill, `^IRX`) via `yfinance`.
- **Data hygiene** — filters stale/illiquid quotes (no two-sided market, crossed
  markets, blown-out spreads, dead open interest) and reports *why* each contract
  was dropped. yfinance data is noisy; this step is load-bearing.
- **Implied-vol solver** — Newton-Raphson fast path with a guaranteed Brent
  fallback on a no-arbitrage bracket; returns `None` for unpriceable quotes
  rather than garbage.
- **Analytic Greeks** — Δ, Γ, vega, Θ, ρ in conventional market units.
- **Model-vs-market divergence** — prices every contract at a single ATM vol and
  measures the gap to market, exposing the skew across OTM strikes.

## Example

```
$ python cli.py SPY
SPY  spot=754.90  expiry=2026-06-23  T=0.022y  r=3.62%  q=0.98%
ATM implied vol (solved): 10.03%
Priceable contracts: 164

Largest flat-vol model vs market divergences across OTM contracts
(the smile a single vol misses):
   put K=753.00   mkt=   5.27  flat-BS=   3.41  diff=  -1.86  IV=14.3%
   put K=750.00   mkt=   4.15  flat-BS=   2.30  diff=  -1.85  IV=14.5%
   ...
```

The OTM put IVs rising as strikes fall — and the flat-vol model consistently
*underpricing* them — is the equity **put skew**, recovered from live quotes.

## Usage

```bash
pip install -r requirements.txt

# Interactive web app (Phase 2): 3D vol surface, smile, term structure, divergence
streamlit run app.py

# Command-line engine (Phase 1)
python cli.py AAPL                 # nearest tradeable expiry
python cli.py SPY --expiry 2026-07-17
python cli.py NVDA --expiries      # list available expiries
python cli.py AAPL --show-drops    # show the data-cleaning report
```

Learn the finance behind it from the runnable teaching notebooks in `notebooks/`:

1. `01_understanding_the_project.ipynb` — options, Black-Scholes, implied vol from scratch
2. `02_volatility_surface.ipynb` — smile, term structure, the full surface
3. `03_monte_carlo.ipynb` — simulation, convergence, Asian/barrier options
4. `04_benchmark.ipynb` — flat-vol vs. SVI vs. ML, head-to-head on live data (**needs internet**)

## Project layout

```
optvol/
  data.py                  live chain pull, cleaning, risk-free rate
  analyze.py               IV solving + Greeks + divergence assembly
  surface.py               multi-expiry surface: smile, term structure, grid
  plots.py                 reusable Plotly figures (surface/smile/term)
  pricing/
    black_scholes.py       vectorized price + analytic Greeks
    implied_vol.py         Newton/Brent IV inversion
    monte_carlo.py         GBM simulation; European/Asian/barrier pricers
  surface_models/
    svi.py                 SVI parametric smile fit + no-arbitrage checks
    ml.py                  gradient-boosted IV(moneyness, T) model
app.py                     Streamlit web app (Phase 2)
cli.py                     command-line entry point
tests/                     put-call parity, IV round-trip, FD-checked Greeks
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

Covers put-call parity, IV round-trip recovery across a strike/vol grid,
no-arbitrage rejection of unpriceable quotes, and finite-difference validation
of analytic vega.

## Roadmap

The project is built in phases, each independently demonstrable and each adding a
distinct, resume-relevant capability. The through-line across every phase is the same
question: **where does the market disagree with the models, and why?**

| Phase | Theme | Status |
|---|---|---|
| 1 | Pricing core, IV inversion, live data, divergence view | ✅ Done |
| 2 | Interactive vol-surface web app | ✅ Done |
| 3 | Monte Carlo & path-dependent options | ✅ Done |
| 4 | Surface modeling (SVI + ML) & benchmark writeup | ✅ Done |
| 5 | LLM market-commentary layer | ⬜ Planned |

---

### ✅ Phase 1 — Pricing core & model-vs-market divergence *(complete)*

The foundation. Replaces a naive single-formula script with a tested package that inverts
Black-Scholes to recover market-implied volatility.

- Vectorized Black-Scholes price + analytic Greeks (Δ, Γ, ν, Θ, ρ)
- Implied-vol solver (Newton fast path → Brent fallback on a no-arbitrage bracket)
- Live data layer: option chains, spot, real risk-free rate (`^IRX`), dividend yield
- Quote-cleaning pipeline with a drop report
- Flat-vol-vs-market divergence that exposes the skew across OTM strikes
- Test suite: put-call parity, IV round-trip, finite-difference Greek checks
- Teaching notebook (`notebooks/01_understanding_the_project.ipynb`)

---

### ✅ Phase 2 — Interactive volatility-surface web app *(complete)*

Turns the engine into a tool people can *use*.

- Streamlit app (`app.py`): type a ticker → live chains pulled, cleaned, and priced
- **3D implied-volatility surface** (moneyness × maturity × IV) across multiple expiries,
  sampled to *span* the term and skip noisy near-0-DTE
- **Smile/skew slices** per expiry and an **ATM term-structure** view
- Sortable **divergence table** (model vs. market, biggest gaps first) + Greeks columns
- Data-quality panel surfacing the cleaning drop report
- Plotting kept in `optvol/plots.py` so the same figures work in app, notebooks, or tests
- 5-minute `st.cache_data` TTL so UI interaction doesn't re-hit yfinance
- Teaching notebook (`notebooks/02_volatility_surface.ipynb`)

**Modules added:** `optvol/surface.py`, `optvol/plots.py`, `app.py`.
**Next deliverable:** deploy to Streamlit Community Cloud + add a hero screenshot here.
**Skills signaled:** turning a model into a product; interactive data viz; messy real-time data;
interpolation of irregular market quotes.

---

### ✅ Phase 3 — Monte Carlo engine & path-dependent options *(complete)*

Goes beyond closed-form Black-Scholes to price payoffs that *have no formula*.

- Vectorized GBM simulator: exact one-step terminal draw + full multi-step paths
- **Antithetic variates** for variance reduction (~65% less variance in the notebook demo)
- Every price returns an `MCResult` with **standard error + 95% confidence interval**
- **Validated** against Black-Scholes: European MC converges to BS with error ∝ 1/√n
  (convergence chart in the notebook; tests assert agreement within 4 standard errors)
- Prices **path-dependent** options BS cannot: **Asian** (arithmetic/geometric average) and
  **barrier** (up/down knock-in/out), with in+out parity checked exactly
- Teaching notebook (`notebooks/03_monte_carlo.ipynb`)

**Module added:** `optvol/pricing/monte_carlo.py`; tests in `tests/test_monte_carlo.py`.
**Skills signaled:** stochastic simulation, risk-neutral pricing by expectation, numerical
convergence analysis, variance-reduction techniques, pricing exotic derivatives.

---

### ✅ Phase 4 — Surface modeling (SVI + ML) & benchmark writeup *(complete)*

The quantitative and ML centerpiece. Models the implied-vol surface itself — the object that
genuinely has *no* closed form — and benchmarks approaches head-to-head on real data.

- **SVI** (Stochastic Volatility Inspired) parametric fit, independently calibrated per expiry
  via constrained least squares; explicit no-arbitrage checks (`b ≥ 0`, `|ρ| < 1`, minimum
  total variance `≥ 0`)
- **ML surface model:** gradient-boosted regressor learning IV as a function of
  `(moneyness, T)` jointly across the whole surface, reporting **cross-validated** error
  (not just optimistic in-sample fit) to honestly quantify overfitting
- **Benchmark notebook** (`notebooks/04_benchmark.ipynb`): flat-vol vs. SVI vs. ML on a live
  surface, with an error table, bar chart, and a written discussion of the SVI/ML trade-off
  (interpretability + arbitrage guarantees vs. raw fit)
- On live SPY data: flat-vol misses the market by **~15 IV points**; SVI and ML both track
  it to **~1–1.3 points** — the concrete, numeric version of the project's thesis

**Modules added:** `optvol/surface_models/svi.py`, `optvol/surface_models/ml.py`.
**Skills signaled:** the finance × ML intersection — parametric vs. learned models, honest
model evaluation (train/CV gap) on noisy real data, arbitrage-aware calibration.

---

### ⬜ Phase 5 — LLM market-commentary layer

Play to LLM strengths most CompFin students don't have. An agent that reads the computed
surface + divergences and writes an analyst-style brief.

**Goals**
- Summarize the surface into structured features (ATM term structure, skew steepness,
  richest/cheapest contracts vs. model, notable divergences)
- LLM generates a grounded **market-commentary brief** from those features — explicitly cited
  to the numbers (no hallucinated figures)
- Wire it into the Phase 2 app as a "Generate commentary" button
- Guardrails: the model only narrates numbers the engine computed

**New modules:** `optvol/commentary.py` (feature extraction + prompt assembly).
**Deliverable:** sample generated briefs committed to the repo; button live in the app.
**Skills signaled:** LLM application design, grounding/anti-hallucination, the finance × NLP edge.

---

### Cross-cutting (ongoing)

- Keep the test suite green as modules are added; add tests per phase
- Cache live data to avoid hammering yfinance during development
- Package metadata (`pyproject.toml`) and CI once Phase 2 lands
- Expand the README hero section with the deployed app link and screenshots
