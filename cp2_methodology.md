
# CP2 Construction Site Selection Methodology

## 1. Overview

The **CP2 Construction Site Selection Project** evaluates the suitability of cities for residential development. The evaluation is based on two major pillars:

- **Profitability (50%)** – Measures economic potential, demand, and feasibility.
- **Hazard Safety (50%)** – Measures the city's safety from natural hazards and physical risks.

The methodology combines these two pillars into a **Final City Score**, used to rank cities for housing development.

## 2. Key Data Inputs

The data required for analysis includes:
- **GRDP** (Gross Regional Domestic Product per capita)
- **Household Consumption** 
- **Population and Household Data**  
- **Employment Data** (from the Labor Force Survey)
- **Housing Prices** (current and forecasted)
- **Family Income**
- **Risk and Hazard Data** (PHIVOLCS, NOAH for earthquake, flood, storm surge, landslides)
- **Amortization Data** (for calculating monthly installments in IPR)

All data is **cleaned**, **normalized**, and **aligned** to the **2024 anchor year**, with projections extending to **2029**.

## 3. Data Cleaning & Preparation

1. **Cleaning**:  
   - Handle missing data through **imputation techniques** or **fallback mechanisms**.
   - Ensure consistent **geographic identifiers** (e.g., `Region`, `Province`, `City` using PSGC codes).

2. **Normalization**:  
   - **Monetary values** (e.g., GRDP, income) are adjusted to **constant 2018 PHP**.
   - **Area data** is standardized, such as population density in **people per km²**.
   - **Hazard risk** data (flood, storm surge, landslide) is scaled into **Low/Medium/High** categories.

3. **Merging**:  
   - Datasets are merged based on the **common keys** (`Region`, `Province`, `City`), with fallback data used if city-level data is unavailable (e.g., use region or province-level data).

## 4. Feature Engineering

1. **Income-to-Payment Ratio (IPR)**:  
   - IPR is calculated using **amortization** for **20-year loans**:
     \[
     	ext{IPR} = rac{	ext{Monthly Household Income}}{	ext{Monthly Housing Installment}}
     \]
   - **Amortization formula** (for 20 years):
     \[
     	ext{Amort\_20yr} = P 	imes rac{r(1 + r)^n}{(1 + r)^n - 1}
     \]
     Where:
     - **P** = Loan amount (house price)
     - **r** = Monthly interest rate
     - **n** = Number of months (240 months for 20 years)

2. **Population and Household Growth**:  
   - **CAGR** (Compound Annual Growth Rate) is used to project future **household** and **population growth** from 2020 data onward.

3. **Hazard Exposure**:  
   - **Flood risk**, **storm surge**, **landslide risk**, and **earthquake proximity** are calculated using **PHIVOLCS** and **NOAH** data.

4. **Housing Unit Capacity**:  
   - Account for the **current number of housing units** and projected **new housing completions** in the coming years.

## 5. Forecasting for 2024–2029

**CAGR** is used to forecast variables over **2024–2029**. The following factors are forecasted:

1. **GRDP per capita**:  
   Forecasted using **CAGR** from 2018–2021 GRDP data.

2. **Household Consumption**:  
   Projected using **CAGR** from the most recent available data.

3. **Employment (Workforce)**:  
   Forecasted using **ETS (Exponential Smoothing State Space Model)** for trend analysis based on **LFS data**.

4. **Income**:  
   Projected using **CAGR** from **2021–2023p** income data.

5. **Housing Prices**:  
   Projected using **driver-based growth models**, incorporating **GRDP** and **household consumption** growth rates.

## 6. Pillar Scoring (Computation)

The scoring for each pillar is computed as follows:

### Economy Pillar (40% of Profitability)
- **GRDP per capita** (forecasted using **CAGR**).
- **Employment rate** (forecasted using **ETS**).
- **Household income** (projected using **CAGR**).

The **Economy Score** is calculated by:
\[
	ext{EconomyScore} = 0.60 	imes 	ext{norm(EmploymentRate)} + 0.40 	imes 	ext{norm(GRDP\_per\_capita)}
\]

### Feasibility Pillar (40% of Profitability)
- **Income-to-Payment Ratio (IPR)**: Based on projected **household income** and **housing prices**.
- **Housing Supply vs. Demand**: Evaluates if the available housing stock can meet projected demand.

The **Feasibility Score** is computed as:
\[
	ext{FeasibilityScore} = 	ext{norm(IPR\_20yr)}
\]

### Demand Pillar (20% of Profitability)
- **Number of Households**: Adjusted for growth using **CAGR**.
- **Occupied Housing Units**: Adjusted for demand and growth.
- **Growth Factor**: Using **CAGR** for household and population growth.

The **Demand Score** is calculated as:
\[
	ext{DemandScore} = 	ext{norm(num\_housing\_units)}
\]

### Profitability Pillar
- **Profitability Score** combines **Economy**, **Feasibility**, and **Demand**:
\[
	ext{ProfitabilityScore} = 0.40 	imes 	ext{EconomyScore} + 0.40 	imes 	ext{FeasibilityScore} + 0.20 	imes 	ext{DemandScore}
\]

### Risk/Hazard Pillar
- **Hazard Safety** combines earthquake, flood, storm surge, and landslide risks. It is computed as:
\[
	ext{HazardSafety\_NoFault} = 1 - (0.40 	imes 	ext{EarthquakeRisk} + 0.40 	imes 	ext{HydroRisk} + 0.20 	imes 	ext{LandslideRisk})
\]

**EarthquakeRisk**, **HydroRisk**, and **LandslideRisk** are calculated based on the normalized values of relevant factors (e.g., earthquake frequency, flood level, storm surge level, etc.).

## 7. Final City Score (Composite Score)

The **Final City Score** combines **Profitability** and **Hazard Safety** into a single ranking:
\[
	ext{FinalCityScore} = 0.50 	imes 	ext{ProfitabilityScore} + 0.50 	imes 	ext{HazardSafety\_NoFault}
\]

## 8. Scenario Analysis

Scenario analysis is run to understand the sensitivity of rankings to changes in **economic factors**, **prices**, and **interest rates**. These scenarios include:
- **BASE** (baseline scenario)
- **RATE +25bp** (interest rate increase)
- **PRICE +10** (increase in housing prices)

Each city’s **FinalCityScore** is recalculated under these different scenarios.

## 9. Final Fact Table Generation

The final fact table includes the following columns:
- **Region**
- **Province**
- **City**
- **Year**
- **Scenario**
- **UnitModel**
- **FeasibilityScore**
- **EconomyScore**
- **DemandScore**
- **ProfitabilityScore**
- **HazardSafetyScore**
- **FinalCityScore**
- **Rank_in_Scenario**
- **Delta_Final_vs_Baseline**

The table is updated for each **scenario** and **unit model** to produce a comprehensive ranking and detailed output.

## 10. Quality Assurance & Validation

1. **Outlier Detection**: Perform checks for outliers in **GRDP** and **income** data.
2. **Coverage Check**: Ensure that all cities are covered, and no regions are missing.
3. **Backtesting**: Validate results by comparing them to actual historical data (if available).

## 11. Python Scripts for Automation

Your **AI agent** should ensure the following scripts are implemented:
1. **Data Ingestion**: Load and preprocess raw datasets.
2. **Preprocessing & Cleaning**: Clean, normalize, and standardize data.
3. **Forecasting**: Apply **CAGR** and **ETS** models to forecast **GRDP**, **income**, **workforce**, and **housing prices**.
4. **Pillar Scoring**: Compute the **Economy**, **Feasibility**, **Demand**, **Profitability**, and **Hazard** scores.
5. **Scenario Analysis**: Perform scenario-based testing and sensitivity analysis.
6. **Fact Table Generation**: Create the final fact table and export to storage.
7. **Export**: Save the final fact table to the appropriate directory (e.g., S3).

## 12. Review of Missing Scripts

Ensure the following critical scripts are included:
- **Feasibility & Economy Score Calculation**: Verify that **IPR** and **GRDP** scores are being calculated correctly.
- **Final Fact Table Generation**: Make sure the script properly assembles all data and calculates the **FinalCityScore**.
- **Scenario Handling**: Double-check that **scenario analysis** scripts are correctly calculating different city scenarios.
- **Amortization Calculation**: Ensure the **monthly installment** (amortization) formula is integrated into IPR.

