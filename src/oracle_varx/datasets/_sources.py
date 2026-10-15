"""Download helpers for the data the package cannot redistribute.

The application in Tan et al. (2026) uses CRSP open-to-close returns of nine SPDR
sector ETFs with ten FRED macro confounders. The CRSP file and the ICE BofA BBB
spread cannot be redistributed, so this module fetches similar data on demand at
three levels of fidelity:

* :func:`fetch_industry_portfolios` — Fama-French daily industry portfolios and
  factors from Ken French's Data Library. This is the source of the dataset shipped
  inside the package (see :func:`oracle_varx.datasets.load_industry`).
* :func:`fetch_sector_etfs` — the paper's own nine sector ETFs and SPY, as
  open-to-close log returns, via ``yfinance``. The closest available replication of
  the paper's application; Yahoo prices are not CRSP prices.
* :func:`fetch_fred` — any FRED series, for the paper's exact confounder presets.

Nothing here is required to use the estimator or to run the shipped example.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "fetch_treasury_curve",
    "fetch_industry_portfolios",
    "fetch_sector_etfs",
    "fetch_fred",
    "SECTOR_ETFS",
    "FRED_PRESETS",
    "TREASURY_NOMINAL",
    "TREASURY_REAL",
]

_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type={kind}"
    "&field_tdr_date_value={year}&page&_format=csv"
)

#: Nominal constant-maturity Treasury yields used as the endogenous system.
TREASURY_NOMINAL: tuple[str, ...] = (
    "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr",
)

#: Real (TIPS) constant-maturity yields used as the observed confounders.
TREASURY_REAL: tuple[str, ...] = ("5 YR", "7 YR", "10 YR", "20 YR", "30 YR")

_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_UA = {"User-Agent": "Mozilla/5.0"}

#: The nine SPDR sector ETFs of Tan et al. (2026) §5.2, plus SPY for the dollar hedge.
SECTOR_ETFS: tuple[str, ...] = (
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU",
)

#: The paper's three confounder presets, as FRED series identifiers (Table 10).
FRED_PRESETS: dict[str, tuple[str, ...]] = {
    "vix": ("VIXCLS",),
    "macro5": ("VIXCLS", "DFF", "T5YIE", "DCOILWTICO", "USEPUINDXD"),
    "all10": (
        "VIXCLS", "DFF", "T5YIE", "DCOILWTICO", "USEPUINDXD",
        "BAMLC0A4CBBB", "DFII10", "DTWEXBGS", "DTWEXEMEGS", "GVZCLS",
    ),
}

_FRENCH_FILES = {
    "industry10": ("10_Industry_Portfolios_daily_CSV.zip", 10),
    "industry12": ("12_Industry_Portfolios_daily_CSV.zip", 12),
    "ff5": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", 6),
    "momentum": ("F-F_Momentum_Factor_daily_CSV.zip", 1),
}


def _download_french(key: str) -> list[list[float | str]]:
    """Download and parse one zipped CSV from Ken French's Data Library."""
    filename, n_col = _FRENCH_FILES[key]
    request = urllib.request.Request(_FRENCH_BASE + filename, headers=_UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    text = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()

    header: list[str] = []
    rows: list[list[float | str]] = []
    for line in text:
        tokens = line.split(",")
        first = tokens[0].strip()
        if not header and line.startswith(",") and len(tokens) >= n_col + 1:
            header = [t.strip() for t in tokens[1 : n_col + 1]]
        # Daily rows are stamped YYYYMMDD; the file also holds annual blocks.
        if len(tokens) >= n_col + 1 and first.isdigit() and len(first) == 8:
            try:
                rows.append([first] + [float(t) for t in tokens[1 : n_col + 1]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No daily rows parsed from {filename}")
    return header or [f"V{i + 1}" for i in range(n_col)], rows


def _french_frame(key: str) -> pd.DataFrame:
    header, rows = _download_french(key)
    frame = pd.DataFrame(rows, columns=["date"] + list(header))
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").sort_index()
    # Ken French publishes percent returns; -99.99 and -999 are missing codes.
    return frame.mask(frame <= -99.0)


def _treasury_year(year: int, kind: str, retries: int = 3) -> pd.DataFrame:
    """Download one calendar year of one Treasury yield curve."""
    url = _TREASURY_URL.format(year=year, kind=kind)
    last: Exception | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            frame = pd.read_csv(io.BytesIO(payload))
            frame["Date"] = pd.to_datetime(frame["Date"], format="mixed")
            return frame.set_index("Date").sort_index()
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
    raise RuntimeError(f"Could not download {kind} for {year}: {last}") from last


def fetch_treasury_curve(
    start_year: int = 2003,
    end_year: int = 2020,
    difference: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the daily US Treasury yield curves, nominal and real.

    This is the source of the dataset shipped with the package. The nominal curve
    is the endogenous system ``Y`` (nine maturities, matching the paper's nine
    series) and the real (TIPS) curve is the observed confounder set ``W``: the
    common real-rate factors that drive every maturity at once and that, left
    unadjusted, manufacture the spurious lagged links the paper warns about.

    Parameters
    ----------
    start_year, end_year : int
        Calendar years to download, inclusive. The real curve begins in 2003.
    difference : bool, default True
        Return first differences in basis points (daily yield changes), which is
        the stationary quantity a VARX should be fitted to. ``False`` returns
        levels in percent.

    Returns
    -------
    (Y, W) : tuple of pandas.DataFrame
        Indexed by date, complete rows only.

    Notes
    -----
    Source: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates
    and Daily Treasury Par Real Yield Curve Rates,
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/.
    A work of the U.S. federal government: **public domain**, which is why this
    dataset — and not the CRSP, Yahoo or Fama-French alternatives — is the one
    shipped inside the package.

    Examples
    --------
    >>> Y, W = fetch_treasury_curve(2019, 2019)  # doctest: +SKIP
    >>> Y.shape[1], W.shape[1]  # doctest: +SKIP
    (9, 5)
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    nominal, real = [], []
    for year in range(start_year, end_year + 1):
        nominal.append(_treasury_year(year, "daily_treasury_yield_curve"))
        real.append(_treasury_year(year, "daily_treasury_real_yield_curve"))

    Y = pd.concat(nominal).sort_index()
    W = pd.concat(real).sort_index()
    Y = Y[[c for c in TREASURY_NOMINAL if c in Y.columns]]
    W = W[[c for c in TREASURY_REAL if c in W.columns]]
    W.columns = [f"real_{c.replace(' ', '')}" for c in W.columns]

    joined = Y.join(W, how="inner").dropna()
    Y, W = joined[Y.columns], joined[W.columns]
    if difference:
        # Basis-point changes: the stationary series the model is fitted to.
        Y = (Y.diff() * 100).dropna()
        W = (W.diff() * 100).loc[Y.index]
    return Y, W


def fetch_industry_portfolios(
    start: str = "2000-01-03",
    end: str = "2020-12-31",
    n_industries: int = 10,
    factors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch daily Fama-French industry portfolios and factors.

    The industry portfolios play the role of the paper's nine sector ETFs, and the
    Fama-French factors play the role of its macro confounders: they are precisely
    the common drivers that, left unobserved, create the spurious lagged links the
    paper warns about in §5.2.

    Parameters
    ----------
    start, end : str
        Inclusive date bounds, any format :func:`pandas.to_datetime` accepts.
    n_industries : {10, 12}, default 10
        Which industry grouping to download.
    factors : bool, default True
        Also download the five Fama-French factors and the momentum factor, and
        return them as the confounders.

    Returns
    -------
    (Y, W) : tuple of pandas.DataFrame
        ``Y`` holds the industry returns in percent, ``W`` the factors (empty with
        ``factors=False``). Both are indexed by date, with complete rows only.

    Notes
    -----
    Source: Kenneth R. French, Data Library,
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.
    Returns are close-to-close, unlike the paper's open-to-close returns.

    Examples
    --------
    >>> Y, W = fetch_industry_portfolios(start="2019-01-01", end="2019-03-31")  # doctest: +SKIP
    >>> Y.shape[1], W.shape[1]  # doctest: +SKIP
    (10, 6)
    """
    if n_industries not in (10, 12):
        raise ValueError(f"n_industries must be 10 or 12, got {n_industries}")

    endogenous = _french_frame(f"industry{n_industries}")
    if factors:
        five = _french_frame("ff5").drop(columns=["RF"], errors="ignore")
        momentum = _french_frame("momentum")
        momentum.columns = ["Mom"]
        confounders = five.join(momentum, how="inner")
    else:
        confounders = pd.DataFrame(index=endogenous.index)

    joined = endogenous.join(confounders, how="inner", lsuffix="_y") if factors else endogenous
    joined = joined.loc[str(start) : str(end)].dropna()
    Y = joined[list(endogenous.columns)]
    W = joined[list(confounders.columns)] if factors else pd.DataFrame(index=joined.index)
    return Y, W


def fetch_sector_etfs(
    start: str = "2000-01-03",
    end: str = "2020-12-31",
    tickers: Sequence[str] = SECTOR_ETFS,
    include_spy: bool = True,
) -> pd.DataFrame:
    """Fetch open-to-close log returns of the paper's sector ETFs via ``yfinance``.

    This is the closest available replication of the application in Tan et al.
    (2026) §5.2, which uses ``log(close / open)`` of the same nine tickers. The
    paper's own prices come from CRSP; Yahoo prices are not identical, so the
    numbers will be close but not equal to the paper's Table 3 and Tables 6-9.

    Parameters
    ----------
    start, end : str
        Inclusive date bounds.
    tickers : sequence of str, default :data:`SECTOR_ETFS`
        The ETFs to fetch.
    include_spy : bool, default True
        Also fetch SPY, needed for the dollar hedge of Appendix C.

    Returns
    -------
    pandas.DataFrame
        Open-to-close log returns, one column per ticker, complete rows only.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed (``pip install "oracle-varx[data]"``).

    Notes
    -----
    Yahoo Finance data is fetched on demand and never redistributed with this
    package. Check Yahoo's terms before using it in published work.

    Examples
    --------
    >>> r = fetch_sector_etfs(start="2020-01-02", end="2020-01-31")  # doctest: +SKIP
    >>> r.columns.tolist()[:3]  # doctest: +SKIP
    ['XLY', 'XLP', 'XLE']
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            'fetch_sector_etfs requires yfinance: pip install "oracle-varx[data]"'
        ) from exc

    symbols = list(tickers) + (["SPY"] if include_spy and "SPY" not in tickers else [])
    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no rows; check the dates and tickers")

    returns = np.log(raw["Close"] / raw["Open"])
    returns = returns[[s for s in symbols if s in returns.columns]]
    returns.columns.name = None
    return returns.dropna()


def fetch_fred(
    series: Sequence[str] | str,
    start: str = "2000-01-03",
    end: str = "2020-12-31",
) -> pd.DataFrame:
    """Fetch one or more series from FRED as a dated frame.

    Use this for the paper's exact confounder presets, which
    :data:`FRED_PRESETS` lists. Two of them are not public domain and are
    deliberately absent from the data shipped with this package: ``VIXCLS`` and
    ``GVZCLS`` are copyright Chicago Board Options Exchange, and ``BAMLC0A4CBBB``
    is an ICE Data Indices product whose full history needs an ICE licence.

    Parameters
    ----------
    series : str or sequence of str
        FRED identifiers, for example ``["DFF", "T5YIE"]``, or a preset name from
        :data:`FRED_PRESETS`.
    start, end : str
        Inclusive date bounds.

    Returns
    -------
    pandas.DataFrame
        One column per series, indexed by date, complete rows only.

    Examples
    --------
    >>> W = fetch_fred("macro5", start="2019-01-01", end="2019-12-31")  # doctest: +SKIP
    >>> W.shape[1]  # doctest: +SKIP
    5
    """
    if isinstance(series, str):
        series = FRED_PRESETS.get(series, (series,))
    frames = {}
    for name in series:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={name}&cosd={start}&coed={end}"
        )
        request = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        frame = pd.read_csv(io.BytesIO(payload))
        frame.columns = ["date", name]
        frame["date"] = pd.to_datetime(frame["date"])
        frames[name] = (
            frame.set_index("date")[name].replace(".", np.nan).astype(float)
        )
    return pd.DataFrame(frames).dropna()
