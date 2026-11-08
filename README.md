# Fixed Income Analytics Terminal

A Python/Streamlit dashboard for bond pricing and interest rate risk analysis, inspired by Bloomberg YAS.

The project focuses on fixed-rate US Treasury bonds, with an independently coded pricing engine and market data retrieved from FRED.

## Features

- **Bond pricing:** Clean/dirty price, YTM, accrued interest and settlement value.
- **Risk analytics:** Modified duration, convexity and DV01.
- **Yield curve:** US Treasury yields (1Y–30Y), 2s10s spread and historical data from FRED.
- **Scenario analysis:** Parallel rate shocks, steepening and flattening scenarios, with estimated P&L.
- **Market data:** Optional bond ETF prices via Yahoo Finance.

Bond characteristics and market prices can be adjusted directly in the dashboard.

## Tech Stack

Python · Streamlit · NumPy · Pandas · Matplotlib · FRED · yfinance (optional)

## Run Locally

```bash
git clone https://github.com/justinearbert/bond-pricing-risk-dashboard.git
cd bond-pricing-risk-dashboard
pip install streamlit pandas numpy matplotlib yfinance
streamlit run app.py
```

## Notes

The pricing engine assumes regular semiannual coupon payments and Treasury-style conventions. Market prices are entered manually; FRED provides published benchmark yields rather than live bond quotes.

Curve scenarios are simplified and intended for analytical purposes, not production trading.

## Author

Justine Arbert  
MSc Financial Markets & Investments — SKEMA Business School
