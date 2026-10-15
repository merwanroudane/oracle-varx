# Data shipped with `oracle-varx`, and how it was pulled

## `treasury_curve.csv.gz` — the shipped real dataset

**What it is.** Daily changes, in basis points, of the US Treasury par yield
curve at nine maturities, together with the real (TIPS) par yield curve at four
maturities.

| Role | Columns |
|---|---|
| Endogenous system `Y` (n = 9) | `3 Mo`, `6 Mo`, `1 Yr`, `2 Yr`, `3 Yr`, `5 Yr`, `7 Yr`, `10 Yr`, `30 Yr` |
| Observed confounders `W` (d = 4) | `real_5YR`, `real_7YR`, `real_10YR`, `real_20YR` |

**Span.** 2006-02-10 to 2020-12-31, 3,724 complete trading days. The start is set
by the 30-year nominal bond, which was discontinued in February 2002 and
reintroduced in February 2006; the 20-year TIPS sets the real-curve constraint.

**Transformation.** Levels are published in percent. The shipped file holds first
differences multiplied by 100, i.e. daily changes in basis points, which is the
stationary quantity a VARX should be fitted to. Rows with any missing maturity are
dropped before differencing, so every row is complete. Values are rounded to one
basis point, which is the publication granularity.

**Source.** U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates
and Daily Treasury Par Real Yield Curve Rates,
<https://home.treasury.gov/resource-center/data-chart-center/interest-rates/>.

**Licence.** A work of the U.S. federal government: **public domain**. This is why
it, and not the alternatives below, is the dataset shipped inside the package.

**Exact pull.** One calendar year per request, both curves, then joined:

```python
from oracle_varx.datasets import fetch_treasury_curve
Y, W = fetch_treasury_curve(start_year=2006, end_year=2020, difference=True)
```

**Why this system.** The paper's application is nine US sector ETFs with ten macro
confounders. The yield curve keeps that shape — nine endogenous series, a handful
of observed common drivers, daily frequency, a multi-thousand-day sample — and
poses the same question: which series lead which, at what lags, once the common
factors are partialled out. The real curve is a genuine confounder set here, since
real-rate shocks move every nominal maturity at once and would otherwise show up
as lead-lag edges along the curve.

---

## Data this package does **not** redistribute

### CRSP open-to-close returns — the paper's own `Y`

`OPCL_20000103_20201231.csv`: daily `log(close / open)` for XLY, XLP, XLE, XLF,
XLV, XLI, XLB, XLK, XLU and SPY, 2000-01-03 to 2020-12-31, from the Center for
Research in Security Prices. **Not redistributable** under the CRSP licence. With
CRSP access, build it from the daily stock file using `OPENPRC` and `|PRC|`.

The closest substitute available without a licence is the same ten tickers from
Yahoo Finance, fetched on demand:

```python
from oracle_varx.datasets import fetch_sector_etfs
returns = fetch_sector_etfs(start="2000-01-03", end="2020-12-31")
```

Yahoo prices are not CRSP prices, so the numbers will be close to but not equal to
the paper's Table 3 and Tables 6-9.

### ICE BofA BBB corporate spread

`BAMLC0A4CBBB`: an ICE Data Indices product. Since April 2026 FRED serves only its
last three years, so the full history needs an ICE licence. Only the paper's
`all10` confounder preset uses it.

### CBOE volatility indices

`VIXCLS` and `GVZCLS` are copyright Chicago Board Options Exchange; the paper's
authors note they were reprinted with permission. They are reachable through
`fetch_fred` but are deliberately absent from the shipped file.

### Fama-French industry portfolios and factors

Reachable, free and standard in finance, and a good equity substitute for the
sector ETFs — but the files carry an explicit copyright line, "Copyright 2026
Eugene F. Fama and Kenneth R. French", so they are fetched on demand rather than
shipped:

```python
from oracle_varx.datasets import fetch_industry_portfolios
Y, W = fetch_industry_portfolios(start="2000-01-03", end="2020-12-31")
```

Ten industry portfolios as the system, the five Fama-French factors plus momentum
as the observed confounders. Returns are close-to-close, not open-to-close.

---

## The synthetic benchmark

`make_oracle_dgp` generates the three-variable time-varying confounded process of
§5.1 and eq. (12) in code, so nothing needs shipping. With its defaults
(`n_obs=3000`, `noise=0.1`, `seed=42`, `legacy_rng=True`) it regenerates the
benchmark data distributed with the paper to floating-point precision:
`max|difference| = 9.6e-16` across all 3,000 rows of both `Y` and `W`.
