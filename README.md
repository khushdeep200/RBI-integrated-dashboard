📊 RBI Financial Stability Analysis using Power BI

📌 Project Overview

This project presents an integrated analysis of the Reserve Bank of India’s (RBI) liquidity operations, cash balances, CRR compliance, and foreign exchange reserves using Microsoft Power BI.
The objective is to transform publicly available RBI data into interactive dashboards that provide insights into monetary policy stance, liquidity conditions, and overall financial stability.
A Composite Financial Stability Score (FSS) is constructed by normalizing and weighting key indicators to summarize the health of the financial system.

🎯 Objectives

Analyze RBI liquidity operations (OMO, Repo, Reverse Repo, SLF)
Assess net liquidity conditions (injection vs absorption)
Evaluate cash balances and CRR compliance
Study foreign exchange reserves trends and composition
Build a composite Financial Stability Score
Present insights through interactive Power BI dashboards

📂 Datasets Used

All datasets are sourced from the Reserve Bank of India (RBI):

Foreign Exchange Reserves
https://ndap.niti.gov.in/dataset/7493

Liquidity Operations (LAF, OMO, SLF)
https://ndap.niti.gov.in/dataset/7499

Cash Balances & CRR Data
https://ndap.niti.gov.in/dataset/7494

Data was exported in CSV format and cleaned using Power BI Power Query.

🧠 Methodology

Data Import & Cleaning
Imported multiple CSV datasets into Power BI
Handled null values and inconsistent formats
Created a unified Calendar table for time analysis
Data Modeling
Established relationships using Year/Month fields
Converted key numeric columns into measures
Derived Metrics:
1. Net Liquidity
2. CRR Compliance Ratio
3. Cash Buffer Ratio
4. Normalized Indicators (0–1 scale)
5. Financial Stability Score (FSS)
Constructed using weighted normalized metrics:
Reserves: 40%
Liquidity Stability: 35%
CRR Compliance: 25%

📊 Dashboards

📌 Dashboard 1 — RBI Liquidity Operations
Repo & Reverse Repo trends
OMO and SLF analysis
Net liquidity injection vs absorption
Liquidity volatility indicators

📌 Dashboard 2 — Cash Balance & CRR Compliance
Actual cash balance vs CRR requirement
CRR compliance KPIs
Cash buffer ratio trends

📌 Dashboard 3 — Foreign Exchange Reserves
USD, Gold, SDR trends
Reserve composition analysis
Indexed reserve growth comparison

📌 Dashboard 4 — Integrated Financial Stability Dashboard
Financial Stability Score trend
Stress / Watch / Stable thresholds
Scatter & bubble analysis
Narrative insights

📈 Key Findings

RBI actively manages liquidity to stabilize financial markets
CRR compliance remains consistently strong
Foreign exchange reserves provide long-term stability
Liquidity volatility is the primary driver of short-term stress
Composite Financial Stability Score simplifies complex macro indicators

🛠 Tools & Technologies

Power BI Desktop
DAX
Power Query
Microsoft Excel / CSV
RBI DBIE Data

📌 Project Structure
├── data/
│   ├── forex_reserves.csv
│   ├── liquidity_operations.csv
│   └── cash_balance_crr.csv
├── dashboards/
│   └── RBI_Financial_Stability.pbix
├── report/
│   └── Project_Report.pdf
└── README.md

🚀 How to Use

Clone the repository
Open RBI_Financial_Stability.pbix in Power BI Desktop
Refresh data (optional)
Explore dashboards using slicers and filters

📌 Limitations

Financial Stability Score is descriptive, not predictive
Some RBI data may have reporting lags
Does not include macro variables like GDP or inflation

🔮 Future Scope

Add machine learning-based stress prediction
Integrate real-time RBI data feeds
Expand analysis with inflation and GDP indicators
Comparative analysis with other economies
