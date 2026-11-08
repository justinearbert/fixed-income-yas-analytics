
"""Fixed Income Analytics Terminal
Developed by Justine Arbert

Run:
    python -m streamlit run app.py

Required:
    streamlit, pandas, numpy, matplotlib

Optional:
    yfinance
"""

from calendar import monthrange
from datetime import date
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fixed Income Analytics Terminal",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# 2. US TREASURY PRICING ENGINE
# ============================================================

def add_months(d, n):
    index = d.year * 12 + d.month - 1 + n
    year, month0 = divmod(index, 12)
    month = month0 + 1

    return date(
        year,
        month,
        min(d.day, monthrange(year, month)[1])
    )


def coupon_schedule(maturity, settlement):
    """
    Regular semiannual coupon schedule,
    anchored to the maturity date.
    """

    if settlement >= maturity:
        raise ValueError(
            "Settlement must be before maturity."
        )

    future = []
    k = 0

    while True:
        coupon_date = add_months(
            maturity,
            -6 * k
        )

        if coupon_date <= settlement:
            previous = coupon_date
            break

        future.append(coupon_date)
        k += 1

        if k > 200:
            raise ValueError(
                "Maturity is too far in the future."
            )

    future.reverse()

    next_date = future[0]

    period_days = (
        next_date - previous
    ).days

    elapsed = (
        settlement - previous
    ).days / period_days

    remaining = (
        next_date - settlement
    ).days / period_days

    return future, elapsed, remaining


def cashflow_data(
    coupon_pct,
    maturity,
    settlement
):
    dates, elapsed, remaining = coupon_schedule(
        maturity,
        settlement
    )

    periods = (
        remaining
        + np.arange(len(dates), dtype=float)
    )

    amounts = np.full(
        len(dates),
        coupon_pct / 2,
        dtype=float
    )

    amounts[-1] += 100.0

    accrued = coupon_pct / 2 * elapsed

    return dates, periods, amounts, accrued


def price_from_yield(
    coupon_pct,
    maturity,
    settlement,
    yield_pct
):
    dates, periods, amounts, accrued = cashflow_data(
        coupon_pct,
        maturity,
        settlement
    )

    base = 1 + yield_pct / 200.0

    if base <= 0:
        raise ValueError(
            "Yield must be above -200%."
        )

    dirty = float(
        np.sum(
            amounts / base ** periods
        )
    )

    clean = dirty - accrued

    return clean, dirty, accrued


def yield_from_clean(
    coupon_pct,
    maturity,
    settlement,
    clean_target
):
    """
    Solve yield to maturity using bisection.
    """

    if clean_target <= 0:
        raise ValueError(
            "Clean price must be positive."
        )

    low = -199.0
    high = 1000.0

    for _ in range(110):

        mid = (low + high) / 2

        clean, _, _ = price_from_yield(
            coupon_pct,
            maturity,
            settlement,
            mid
        )

        if clean > clean_target:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def treasury_risk(
    coupon_pct,
    maturity,
    settlement,
    yield_pct,
    nominal
):
    """
    Calculate:
    - Clean price
    - Dirty price
    - Accrued interest
    - Modified duration
    - Convexity
    - DV01
    - Settlement amount
    """

    dates, periods, amounts, accrued = cashflow_data(
        coupon_pct,
        maturity,
        settlement
    )

    base = 1 + yield_pct / 200.0

    if base <= 0:
        raise ValueError("Invalid yield.")

    pv = amounts / base ** periods

    dirty = float(pv.sum())

    clean = dirty - accrued

    modified = float(
        np.sum(
            pv * periods / (2 * base)
        ) / dirty
    )

    convexity = float(
        np.sum(
            pv
            * periods
            * (periods + 1)
            / (4 * base ** 2)
        ) / dirty
    )

    # Finite-difference DV01: +/- 1 bp

    down = price_from_yield(
        coupon_pct,
        maturity,
        settlement,
        yield_pct - 0.01
    )[0]

    up = price_from_yield(
        coupon_pct,
        maturity,
        settlement,
        yield_pct + 0.01
    )[0]

    dv01 = (
        (down - up)
        / 2
        * nominal / 100
    )

    return {
        "clean": clean,
        "dirty": dirty,
        "accrued": accrued,
        "modified": modified,
        "convexity": convexity,
        "dv01": dv01,
        "principal": clean * nominal / 100,
        "accrued_cash": accrued * nominal / 100,
        "settlement_cash": dirty * nominal / 100,
        "next_coupon": dates[0],
        "dates": dates,
        "periods": periods,
        "cashflows": amounts
    }


# ============================================================
# 3. TREASURY QUOTE CONVERSION
# ============================================================

def parse_treasury_quote(s):
    """
    Convert Treasury quotes into decimal prices.

    Example:
    95-052 = 95 + 5/32 + 2/256
    """

    match = re.fullmatch(
        r"\s*(\d+)-(\d{2})([0-7+])?\s*",
        s
    )

    if not match:
        raise ValueError(
            "Use a quote such as 95-052, "
            "95-05+ or 95-05."
        )

    whole, thirtyseconds, sub = match.groups()

    if int(thirtyseconds) > 31:
        raise ValueError(
            "32nds must be between 00 and 31."
        )

    eighths = (
        4 if sub == "+"
        else int(sub or 0)
    )

    return (
        int(whole)
        + int(thirtyseconds) / 32
        + eighths / 256
    )


# ============================================================
# 4. NON-PARALLEL YIELD SHOCK ENGINE
# ============================================================

def reprice_curve_shock(
    metrics,
    base_yield_pct,
    nominal,
    short_bp,
    long_bp
):
    """
    Illustrative tenor-dependent yield shocks.

    Short-end shock: 2Y
    Long-end shock: 10Y

    Interpolation is applied across cash-flow tenors.
    This is NOT a calibrated zero-coupon curve model.
    """

    tenors = metrics["periods"] / 2.0

    shocks_bp = np.interp(
        tenors,
        [2, 10],
        [short_bp, long_bp]
    )

    shifted_yields = (
        base_yield_pct
        + shocks_bp / 100.0
    )

    bases = (
        1 + shifted_yields / 200.0
    )

    if np.any(bases <= 0):
        raise ValueError(
            "Shocked yields are outside model bounds."
        )

    dirty = float(
        np.sum(
            metrics["cashflows"]
            / bases ** metrics["periods"]
        )
    )

    pnl = (
        dirty - metrics["dirty"]
    ) * nominal / 100

    return pnl


# ============================================================
# 5. FRED MARKET DATA
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def fred_treasury_yields():

    tickers = [
        "DGS1",
        "DGS2",
        "DGS3",
        "DGS5",
        "DGS7",
        "DGS10",
        "DGS20",
        "DGS30"
    ]

    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id="
        + ",".join(tickers)
    )

    data = pd.read_csv(url)

    data = data.rename(
        columns={
            data.columns[0]: "Date"
        }
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    for ticker in tickers:
        data[ticker] = pd.to_numeric(
            data[ticker],
            errors="coerce"
        )

    return (
        data
        .dropna(subset=["Date"])
        .set_index("Date")
        .sort_index()
    )


# ============================================================
# 6. OPTIONAL YAHOO FINANCE DATA
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def yahoo_etf_prices():

    import yfinance as yf

    symbols = [
        "SHY",
        "IEF",
        "TLT",
        "LQD"
    ]

    data = yf.download(
        symbols,
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "No Yahoo Finance data returned."
        )

    return (
        data["Close"]
        .dropna(how="all")
    )


# ============================================================
# 7. MAIN INTERFACE
# ============================================================

st.title("Fixed Income Analytics Terminal")

st.caption(
    "Developed by Justine Arbert | "
    "MSc Financial Markets & Investments | "
    "SKEMA Business School"
)

st.markdown("""
### Treasury Pricing, Interest Rate Risk & Market Analytics

This interactive application provides an independent
framework for analysing fixed-rate US Treasury securities.

It combines:

- Bond valuation and yield calculations.
- Duration, convexity and DV01.
- Cash-flow and settlement analysis.
- Automatically refreshed public Treasury yield data.
- Parallel and non-parallel interest-rate scenarios.

The application is designed for educational purposes
and Fixed Income Sales-oriented market analysis.
""")

st.divider()


# ============================================================
# 8. SIDEBAR — BOND ASSUMPTIONS
# ============================================================

with st.sidebar:

    st.header("Bond Inputs")

    st.caption(
        "US Treasury-style fixed-rate bond "
        "with semiannual coupons."
    )

    settlement = st.date_input(
        "Settlement date",
        value=date.today()
    )

    maturity = st.date_input(
        "Maturity date",
        value=date(2036, 8, 15)
    )

    coupon = st.number_input(
        "Annual coupon (%)",
        min_value=0.0,
        max_value=25.0,
        value=4.625,
        step=0.125
    )

    nominal = st.number_input(
        "Position nominal (USD)",
        min_value=100.0,
        value=1_000_000.0,
        step=100_000.0
    )

    st.divider()

    st.subheader("Market Assumption")

    input_mode = st.radio(
        "Select input",
        [
            "Clean price",
            "Yield to maturity"
        ]
    )

    if input_mode == "Clean price":

        clean_input = st.number_input(
            "Clean price per $100",
            min_value=0.01,
            value=100.0,
            step=0.125,
            format="%.6f"
        )

        use_quote = st.checkbox(
            "Enter US Treasury 32nds quote instead"
        )

        if use_quote:

            quote = st.text_input(
                "Treasury quote",
                value="100-00"
            )

    else:

        yield_input = st.number_input(
            "Yield to maturity (%)",
            min_value=-10.0,
            max_value=50.0,
            value=4.50,
            step=0.05
        )

    st.caption(
        "Prices and yields are user-entered, "
        "not live quotes for an individual bond."
    )


# ============================================================
# 9. CALCULATE BOND ANALYTICS
# ============================================================

try:

    if input_mode == "Clean price":

        if use_quote:
            selected_clean = parse_treasury_quote(
                quote
            )
        else:
            selected_clean = clean_input

        ytm = yield_from_clean(
            coupon,
            maturity,
            settlement,
            selected_clean
        )

    else:
        ytm = yield_input

    metrics = treasury_risk(
        coupon,
        maturity,
        settlement,
        ytm,
        nominal
    )

    valid_bond = True

except (ValueError, OverflowError) as exc:

    st.error(
        f"Please check the bond inputs: {exc}"
    )

    valid_bond = False


# ============================================================
# 10. DASHBOARD TABS
# ============================================================

pricing_tab, curve_tab, scenario_tab, methods_tab = st.tabs([
    "Bond Pricing & Risk",
    "Treasury Yield Curve",
    "Scenario Analysis",
    "Methodology"
])


# ============================================================
# 11. TAB 1 — BOND PRICING & RISK
# ============================================================

with pricing_tab:

    st.subheader(
        "Bond Valuation & Interest Rate Risk"
    )

    if valid_bond:

        cols = st.columns(4)

        cols[0].metric(
            "Clean Price / 100",
            f"{metrics['clean']:.4f}"
        )

        cols[1].metric(
            "Dirty Price / 100",
            f"{metrics['dirty']:.4f}"
        )

        cols[2].metric(
            "Accrued Interest / 100",
            f"{metrics['accrued']:.4f}"
        )

        cols[3].metric(
            "Yield to Maturity",
            f"{ytm:.4f}%"
        )

        st.divider()

        cols = st.columns(4)

        cols[0].metric(
            "Modified Duration",
            f"{metrics['modified']:.3f} years"
        )

        cols[1].metric(
            "Convexity",
            f"{metrics['convexity']:.2f}"
        )

        cols[2].metric(
            "DV01 (Position)",
            f"${metrics['dv01']:,.2f}"
        )

        cols[3].metric(
            "Settlement Amount",
            f"${metrics['settlement_cash']:,.2f}"
        )

        # ----------------------------------------
        # Settlement breakdown
        # ----------------------------------------

        with st.expander(
            "Settlement Breakdown & Coupon Dates"
        ):

            st.write(
                f"Principal (clean): "
                f"**${metrics['principal']:,.2f}**"
            )

            st.write(
                f"Accrued interest: "
                f"**${metrics['accrued_cash']:,.2f}**"
            )

            st.write(
                f"Total settlement: "
                f"**${metrics['settlement_cash']:,.2f}**"
            )

            st.write(
                f"Next coupon: "
                f"**{metrics['next_coupon']:%d %B %Y}**"
            )

            cf_table = pd.DataFrame({
                "Payment date": metrics["dates"],
                "Cash flow per $100": metrics["cashflows"]
            })

            st.dataframe(
                cf_table,
                hide_index=True,
                use_container_width=True
            )

        # ----------------------------------------
        # Price-yield relationship
        # ----------------------------------------

        st.markdown(
            "### Bond Price–Yield Relationship"
        )

        yield_grid = np.linspace(
            max(-10, ytm - 3),
            ytm + 3,
            121
        )

        prices = [
            price_from_yield(
                coupon,
                maturity,
                settlement,
                float(y)
            )[0]
            for y in yield_grid
        ]

        fig, ax = plt.subplots(
            figsize=(8, 3.5)
        )

        ax.plot(
            yield_grid,
            prices
        )

        ax.scatter(
            [ytm],
            [metrics["clean"]],
            zorder=3
        )

        ax.set_xlabel(
            "Yield to Maturity (%)"
        )

        ax.set_ylabel(
            "Clean Price per $100"
        )

        ax.set_title(
            "Bond Price vs Yield"
        )

        ax.grid(alpha=0.2)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------
        # Sales interpretation
        # ----------------------------------------

        st.info(
            f"For a ${nominal:,.0f} nominal position, "
            f"a +1 bp yield move implies approximately "
            f"**-${metrics['dv01']:,.2f}** in price P&L.\n\n"
            "This is a local interest-rate sensitivity, "
            "not a guaranteed investment return."
        )


# ============================================================
# 12. TAB 2 — TREASURY YIELD CURVE
# ============================================================

with curve_tab:

    st.subheader(
        "US Treasury Yield Curve — FRED"
    )

    st.caption(
        "Automatically updated to the latest available "
        "daily observations. Not live intraday prices."
    )

    if st.button(
        "Refresh Public Market Data"
    ):

        fred_treasury_yields.clear()
        yahoo_etf_prices.clear()

    try:

        data = fred_treasury_yields()

        tickers = [
            "DGS1",
            "DGS2",
            "DGS3",
            "DGS5",
            "DGS7",
            "DGS10",
            "DGS20",
            "DGS30"
        ]

        complete = data.dropna(
            subset=tickers
        )

        if complete.empty:
            raise ValueError(
                "No complete Treasury curve "
                "observation available."
            )

        latest = complete.iloc[-1]

        observation_date = complete.index[-1]

        st.caption(
            f"Curve observation date: "
            f"{observation_date:%d %B %Y} | "
            "Source: FRED / US Treasury"
        )

        # ----------------------------------------
        # Market indicators
        # ----------------------------------------

        cols = st.columns(3)

        cols[0].metric(
            "2Y Treasury",
            f"{latest['DGS2']:.2f}%"
        )

        cols[1].metric(
            "10Y Treasury",
            f"{latest['DGS10']:.2f}%"
        )

        spread_10y_2y = (
            latest["DGS10"]
            - latest["DGS2"]
        ) * 100

        cols[2].metric(
            "10Y - 2Y Spread",
            f"{spread_10y_2y:+.1f} bp"
        )

        # ----------------------------------------
        # Yield curve
        # ----------------------------------------

        maturities = [
            1, 2, 3, 5, 7, 10, 20, 30
        ]

        yields = [
            float(latest[t])
            for t in tickers
        ]

        fig, ax = plt.subplots(
            figsize=(8, 3.5)
        )

        ax.plot(
            maturities,
            yields,
            marker="o"
        )

        ax.set_xlabel(
            "Maturity (years)"
        )

        ax.set_ylabel(
            "Yield (%)"
        )

        ax.set_title(
            "US Treasury Constant-Maturity Yield Curve"
        )

        ax.grid(alpha=0.2)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------
        # Historical yield spread
        # ----------------------------------------

        st.markdown(
            "### Historical 10Y–2Y Treasury Spread"
        )

        history = (
            data[["DGS2", "DGS10"]]
            .dropna()
            .tail(260)
        )

        spread_history = pd.DataFrame({
            "10Y - 2Y (bp)": (
                history["DGS10"]
                - history["DGS2"]
            ) * 100
        })

        st.line_chart(
            spread_history
        )

        # ----------------------------------------
        # Yield curve data table
        # ----------------------------------------

        curve_table = pd.DataFrame({
            "Maturity": [
                f"{n}Y"
                for n in maturities
            ],
            "Yield (%)": yields
        })

        st.dataframe(
            curve_table,
            hide_index=True,
            use_container_width=True
        )

    except Exception as exc:

        st.warning(
            f"FRED data unavailable: {exc}"
        )

        st.caption(
            "Bond pricing and scenario analysis "
            "remain available without internet."
        )

    # ----------------------------------------
    # Yahoo Finance ETF monitor
    # ----------------------------------------

    st.divider()

    st.markdown(
        "### Bond ETF Market Monitor"
    )

    st.caption(
        "Yahoo Finance closing prices: "
        "SHY, IEF, TLT and LQD."
    )

    if st.checkbox(
        "Load Yahoo Finance ETF Prices"
    ):

        try:

            etf_data = yahoo_etf_prices()

            st.line_chart(
                etf_data
            )

        except ImportError:

            st.warning(
                "Install yfinance using: "
                "python -m pip install yfinance"
            )

        except Exception as exc:

            st.warning(
                f"Yahoo Finance unavailable: {exc}"
            )


# ============================================================
# 13. TAB 3 — SCENARIO ANALYSIS
# ============================================================

with scenario_tab:

    st.subheader(
        "Interest Rate Scenario Analysis"
    )

    if valid_bond:

        # ----------------------------------------
        # Parallel yield shocks
        # ----------------------------------------

        st.markdown(
            "### 1. Parallel Yield Shock"
        )

        bp = st.slider(
            "Parallel yield move (basis points)",
            min_value=-200,
            max_value=200,
            value=35,
            step=5
        )

        shifted_yield = (
            ytm + bp / 100
        )

        shifted_clean = price_from_yield(
            coupon,
            maturity,
            settlement,
            shifted_yield
        )[0]

        exact_pnl = (
            shifted_clean
            - metrics["clean"]
        ) * nominal / 100

        approx_pnl = (
            -metrics["dv01"] * bp
            + 0.5
            * metrics["convexity"]
            * metrics["dirty"]
            * (bp / 10000) ** 2
            * nominal / 100
        )

        cols = st.columns(3)

        cols[0].metric(
            "Shocked YTM",
            f"{shifted_yield:.3f}%"
        )

        cols[1].metric(
            "Repriced P&L",
            f"${exact_pnl:+,.2f}"
        )

        cols[2].metric(
            "Duration + Convexity P&L",
            f"${approx_pnl:+,.2f}"
        )

        st.caption(
            "Repriced P&L uses full cash-flow repricing "
            "within this model. It is not a forecast "
            "of realized market P&L."
        )

        st.divider()

        # ----------------------------------------
        # Non-parallel curve scenarios
        # ----------------------------------------

        st.markdown(
            "### 2. Non-Parallel Curve Scenarios"
        )

        st.write(
            "Explore how different movements in "
            "short-term and long-term yields affect "
            "the bond's theoretical valuation."
        )

        presets = {
            "Custom": (0, 0),
            "Bear steepener": (25, 75),
            "Bull steepener": (-75, -25),
            "Bear flattener": (75, 25),
            "Bull flattener": (-25, -75)
        }

        choice = st.selectbox(
            "Select market scenario",
            list(presets)
        )

        short_default, long_default = presets[choice]

        cols = st.columns(2)

        with cols[0]:

            short_bp = st.number_input(
                "2Y Yield Shock (bp)",
                value=short_default,
                step=5,
                key=f"short_{choice}"
            )

        with cols[1]:

            long_bp = st.number_input(
                "10Y Yield Shock (bp)",
                value=long_default,
                step=5,
                key=f"long_{choice}"
            )

        pnl_curve = reprice_curve_shock(
            metrics,
            ytm,
            nominal,
            short_bp,
            long_bp
        )

        st.metric(
            "Illustrative Curve-Shock P&L",
            f"${pnl_curve:+,.2f}"
        )

        # ----------------------------------------
        # Shock curve visualization
        # ----------------------------------------

        shock_df = pd.DataFrame({
            "Tenor (years)": [
                0, 2, 5, 10, 20, 30
            ],
            "Yield Shock (bp)": [
                short_bp,
                short_bp,
                (5 * short_bp + 3 * long_bp) / 8,
                long_bp,
                long_bp,
                long_bp
            ]
        })

        st.line_chart(
            shock_df.set_index(
                "Tenor (years)"
            )
        )

        st.caption(
            "Simplified scenario: the model starts from "
            "a flat yield equal to the bond's YTM, "
            "then interpolates the 2Y and 10Y shocks "
            "across cash-flow tenors. "
            "This is not a bootstrapped zero curve "
            "or a trading-grade risk model."
        )


# ============================================================
# 14. TAB 4 — METHODOLOGY
# ============================================================

with methods_tab:

    st.subheader(
        "Model Assumptions & Limitations"
    )

    st.markdown("""
### Pricing Methodology

The application uses an independently coded
fixed-income valuation framework.

**Supported instruments:**

- Fixed-coupon US Treasury-style securities.
- Semiannual coupon payments.
- Regular coupon schedules.
- Non-callable securities.

**Pricing calculations:**

- Clean price.
- Dirty price.
- Accrued interest.
- Yield to maturity.
- Modified duration.
- Convexity.
- DV01.
- Settlement amount.

### Market Data

**FRED**

Provides published daily US Treasury
constant-maturity yields.

These are benchmark market yields,
not prices for individual Treasury bonds.

**Yahoo Finance**

Provides optional historical closing
prices for selected bond ETFs.

Data may be delayed or unavailable.

### Scenario Analysis

Parallel yield shocks are evaluated
through cash-flow repricing.

Duration and convexity are used to
approximate changes in bond value.

Non-parallel yield scenarios are
simplified and illustrative.

They do not use a fully calibrated
zero-coupon or forward-rate curve.

### Limitations

- No Bloomberg API connection.
- No guaranteed real-time prices.
- No live pricing of individual Treasury securities.
- No credit or default model.
- No transaction costs or bid-ask spreads.
- No callable bond pricing.
- No embedded options.
- No bootstrapped Z-spread or OAS model.
- No support for irregular coupon schedules.
- Not designed for all corporate bond conventions.

The application is intended for
educational and analytical purposes only.

It does not constitute investment advice.
""")


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Fixed Income Analytics Terminal | "
    "Python · Streamlit · Fixed Income Markets"
)
