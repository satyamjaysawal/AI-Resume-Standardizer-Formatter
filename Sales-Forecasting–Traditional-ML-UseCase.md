





# Complete Mermaid Diagram for Sales Forecasting Project
```mermaid
flowchart TD
    %% Start of the process
    Start([Start Sales Forecasting Project]) --> Setup[Project Setup Phase]
   
    %% Setup Phase
    subgraph SetupPhase [Initial Setup]
        direction TB
        S1[Create Folder Structure]
        S2[Install Python Dependencies<br>pip install -r requirements.txt]
        S3[Prepare Input Data<br>Place Sales forecasting train data.csv & test data.csv in input/]
        S4[Run Preprocessing: python preprocess.py<br>Generates sales_data.csv]
        S5[Optional: Segment Customers<br>Identify New/Churned/Irregular]
    end
   
    Setup --> SetupPhase
    SetupPhase --> EDA[EDA Phase]
   
    %% EDA Phase
    subgraph EDAPhase [Exploratory Data Analysis]
        direction TB
        E1[Load Data from input/sales_data.csv]
        E2[Aggregate to Monthly Level]
        E3[Visualize Trends, Seasonality, Anomalies]
        E4[Analyze Customer & Plant Behaviors]
        E5[Detect Outliers & Handle Missing Data]
        E6[Bonus: Classify Customers (New/Churned/Irregular)]
    end
   
    EDA --> EDAPhase
    EDAPhase --> FeatureEng[Feature Engineering Phase]
   
    %% Feature Engineering Phase
    subgraph FeatureEngPhase [Feature Engineering]
        direction TB
        F1[Create Time-Based Features (Lags, Rolling Stats)]
        F2[Encode Categorical Variables (Customer, Plant)]
        F3[Handle Seasonality (Fourier Terms, Dummies)]
        F4[Segment Data by Customer & Plant]
        F5[Bonus: Custom Features for Irregular Customers]
    end
   
    FeatureEng --> FeatureEngPhase
    FeatureEngPhase --> Model[Modeling Phase]
   
    %% Modeling Phase
    subgraph ModelPhase [Model Training & Forecasting]
        direction TB
        M1[Split Data into Train/Test]
        M2[Train Models (e.g., XGBoost, Prophet)]
        M3[Evaluate with MAPE, RMSE]
        M4[Forecast Next 12 Months per Customer/Plant]
        M5[Aggregate to Plant Level & Validate >=80% Accuracy]
        M6[Bonus: Specialized Models for Customer Types]
    end
   
    Model --> ModelPhase
    ModelPhase --> Output[Save Forecasts to output/forecasts.csv]
   
    %% Validation Phase
    Output --> Validate[Validation Phase]
   
    subgraph ValidatePhase [Output Validation]
        direction TB
        V1[Compare Forecasts vs. Test Data]
        V2[Visualize Predictions & Errors]
        V3[Check Plant-Level Accuracy (>=80%)]
        V4[Review Customer Segments & Bonus Classifications]
    end
   
    Validate --> ValidatePhase
    ValidatePhase --> Decision{Is Accuracy Satisfactory?}
   
    %% Decision Paths
    Decision -->|No| Adjust[Adjust Features or Models]
    Adjust --> UpdateModels[Retune Hyperparameters or Resegment]
    UpdateModels --> FeatureEng
   
    Decision -->|Yes| Success[Success! Process Complete]
    Success --> FinalOutput[Final Output: Monthly Forecasts for 12 Months]
   
    %% External Files
    InputFile[sales_data.csv Input Data]
   
    S3 -.-> InputFile
   
    %% Styling
    style Start fill:#4CAF50,color:white
    style Success fill:#4CAF50,color:white
    style FinalOutput fill:#4CAF50,color:white
   
    classDef phase fill:#E1F5FE,stroke:#01579B,stroke-width:2px
    class SetupPhase,EDAPhase,FeatureEngPhase,ModelPhase,ValidatePhase phase
```
****
# Complete Scenario of the Sales Forecasting Project
## Overall Scenario (What the Project Does):
The requirements (what you need to set it up), and finally how to achieve the solution (step-by-step implementation and execution). This project is designed to forecast sales amount ($) and quantity (LBS) for the next 12 months (Oct 2025–Sep 2026) at customer and plant levels using traditional ML techniques, based on historical sales data.
#### 1. Project Scenario: What It Does and Why
- **Problem it Solves**: Sales forecasting is critical for inventory management, supply chain optimization, and financial planning in manufacturing or distribution businesses. With sparse or irregular data (e.g., new/churned customers), accurate predictions are challenging. This project addresses that by:
  - Preprocessing weekly train/test data into monthly aggregated format.
  - Performing EDA to uncover trends, seasonality, and customer/plant patterns.
  - Engineering features and building models to forecast monthly sales/quantity.
  - Generating predictions at granular (customer) and aggregated (plant) levels.
  - Bonus: Identifying and handling new, churned, and irregular customers using order history patterns.
- **Use Case Example**: A company with multiple plants selling to customers needs to predict demand to avoid stockouts or overproduction. The tool processes historical data, outputs 12-month forecasts from Oct 2025 onward, and flags customer types for targeted strategies (e.g., promotions for churned customers).
- **Key Workflow**:
  1. Input: Weekly sales data (train/test CSVs).
  2. Preprocessing: Combine, aggregate to monthly.
  3. Processing: EDA → Feature engineering → Model training → Forecasting.
  4. Output: CSV with monthly forecasts for each (plant, customer) pair, plus visualizations.
- **Benefits**: Improves accuracy (aiming >=80% at plant level), handles data sparsity, reproducible, and extensible for bonus features.
- **Limitations in Scenario**: Assumes monthly aggregation; may underperform on very sparse data; no real-time integration; requires sufficient historical data. Current date is September 01, 2025, so forecasts start from October 2025.
This scenario fits real-world supply chain and retail forecasting needs.
#### 2. Requirements: What You Need
To run this project, you'll need hardware, software, services, and files. Here's a breakdown:
- **Hardware/Environment**:
  - A computer with Python 3.8+ installed.
  - Sufficient RAM/CPU for ML training (e.g., 8GB+ RAM recommended).
- **Software Dependencies** (Listed in `requirements.txt`):
  - pandas: For data manipulation.
  - numpy: For numerical operations.
  - matplotlib, seaborn: For visualizations.
  - scikit-learn: For ML models and metrics.
  - xgboost: For gradient boosting models.
  - prophet: For time series forecasting.
  - statsmodels: For statistical tests.
  Install with: `pip install -r requirements.txt`.
- **Files and Folders**:
  - `input/Sales forecasting train data.csv` & `input/Sales forecasting test data.csv`: Historical data from images (transcribe fully).
  - `input/sales_data.csv`: Generated by preprocess.py.
  - `output/forecasts.csv`: Generated forecasts.
  - `output/plots/`: For EDA and prediction visualizations.
- **Skills/Knowledge**:
  - Basic Python and data science understanding.
  - Familiarity with time series and ML concepts.
No external services (e.g., Azure) are required; it's all local.
#### 3. How to Achieve the Solution: Step-by-Step Guide
Here's how the project works under the hood and how you can set it up, run it, and customize it. Use `preprocess.py` first, then `main.py`.
- **Step 1: Project Setup**
  - Create the folder structure as shown.
  - Install dependencies: `pip install -r requirements.txt`.
  - Place train/test CSVs in `input/` (transcribe from images).
- **Step 2: Preprocessing**
  - Execute: `python preprocess.py`.
  - What Happens: Combines train/test, selects columns, aggregates to monthly, saves `sales_data.csv`.
- **Step 3: Running the Forecasting**
  - Execute: `python main.py`.
  - What Happens Internally (Code Flow with Comments Reference):
    - Loads monthly data.
    - Performs EDA (plots trends).
    - Engineers features (lags, rolling means, seasonality).
    - Trains XGBoost models.
    - Forecasts 12 months ahead from Oct 2025.
    - Saves outputs and visualizations.
- **Step 4: Achieving Customization and Enhancements**
  - **Adjust Models**: Add Prophet per group for better seasonality.
  - **Handle Sparsity**: Use more lags or hierarchical forecasting.
  - **Bonus Features**: Integrate customer classification (add to main.py if needed).
  - **Error Handling**: Add checks for missing data.
  - **Scaling**: For large data, use Dask; integrate with databases.
  - **Testing**: Use holdout set; aim for plant-level accuracy >=80%.
****
### Complete Sales Forecasting Project with Step-by-Step Comments
Below is the full "sales-forecasting" project. I've added detailed # comments to **preprocess.py** and **main.py** explaining each step. The code handles weekly to monthly aggregation; uses XGBoost; forecasts from Oct 2025–Sep 2026. Bonus classification can be added to main.py. Transcribe full image data to CSVs before running.
#### Folder Structure
```
sales-forecasting/
├── requirements.txt # Dependencies to install via pip
├── input/ # Folder for input data
│ ├── Sales forecasting train data.csv # Transcribe train image
│ ├── Sales forecasting test data.csv # Transcribe test image
│ └── sales_data.csv # Generated by preprocess.py
├── output/ # Folder for generated files (created automatically)
│ ├── forecasts.csv # Output forecasts
│ └── plots/ # EDA and prediction plots
├── preprocess.py # Prepares train+test → sales_data.csv
└── main.py # Main script with forecasting pipeline
```
#### 1. requirements.txt
```
pandas==2.2.2
numpy==2.1.0
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.1
xgboost==2.1.1
prophet==1.1.5
statsmodels==0.14.2
```
#### 2. preprocess.py (With # Comments Explaining Each Step)
```python
import pandas as pd
import os

# Step 0: Define paths
# Input train/test CSVs from images; output combined monthly data
train_path = "input/Sales forecasting train data.csv"
test_path = "input/Sales forecasting test data.csv"
output_path = "input/sales_data.csv"

# Step 1: Load train and test data
# Handle any parsing errors; assume CSV format from images
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Step 2: Select and rename relevant columns
# Based on image headers: WeeklySalesDate -> date, ParentCustomerId -> customer_id, etc.
cols = {
    "WeeklySalesDate": "date",
    "ParentCustomerId": "customer_id",
    "ParentPlantName": "plant_id",
    "ActualSalesAmt": "amount",
    "ActualSalesWt": "quantity",
}
train_df = train_df[list(cols.keys())].rename(columns=cols)
test_df = test_df[list(cols.keys())].rename(columns=cols)

# Step 3: Combine train and test into one dataframe
# For full historical data
df = pd.concat([train_df, test_df], ignore_index=True)

# Step 4: Convert date column to datetime
# Handle any format issues
df["date"] = pd.to_datetime(df["date"])

# Step 5: Aggregate to monthly level
# Group by month, customer, plant; sum amount and quantity
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
monthly_df = (
    df.groupby(["month", "customer_id", "plant_id"])
    .agg({"amount": "sum", "quantity": "sum"})
    .reset_index()
)

# Step 6: Save preprocessed data
# For use in main.py
os.makedirs("input", exist_ok=True)
monthly_df.to_csv(output_path, index=False)
print(f"✅ Preprocessed file saved to {output_path}")
```
#### 3. main.py (With # Comments Explaining Each Step)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Step 0: Set up paths and parameters
# Current date: September 01, 2025; forecast from October 2025
input_path = "input/sales_data.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
forecast_horizon = 12
current_date = datetime(2025, 9, 1)
forecast_start = current_date + relativedelta(months=1)

# Step 1: Load preprocessed data
# From preprocess.py; monthly aggregated
df = pd.read_csv(input_path, parse_dates=["month"])
print("✅ Data Loaded:", df.shape)

# Step 2: Exploratory Data Analysis (EDA)
# Visualize trends by plant
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="month", y="amount", hue="plant_id")
plt.title("Sales Amount Trends by Plant")
plt.savefig(os.path.join(output_dir, "plots", "amount_trends.png"))
plt.close()

# Seasonality (monthly averages)
monthly_avg = df.groupby(df['month'].dt.month).agg({'amount': 'mean', 'quantity': 'mean'})
plt.figure(figsize=(8, 4))
sns.barplot(x=monthly_avg.index, y='amount', data=monthly_avg)
plt.title('Average Monthly Sales Amount (Seasonality)')
plt.savefig(os.path.join(output_dir, "plots", "seasonality.png"))
plt.close()

# Bonus: Customer classification (add if needed; omitted for simplicity in this version)

# Step 3: Feature Engineering
# Add lags, rolling means, seasonality terms
def add_features(group):
    group = group.sort_values("month")
    for lag in [1, 3, 6, 12]:
        group[f"amount_lag_{lag}"] = group["amount"].shift(lag)
        group[f"quantity_lag_{lag}"] = group["quantity"].shift(lag)
    group["amount_roll3"] = group["amount"].rolling(3).mean().shift(1)
    group["quantity_roll3"] = group["quantity"].rolling(3).mean().shift(1)
    group["month_sin"] = np.sin(2 * np.pi * group["month"].dt.month / 12)
    group["month_cos"] = np.cos(2 * np.pi * group["month"].dt.month / 12)
    return group

grouped = df.groupby(["customer_id", "plant_id"])
featured_df = grouped.apply(add_features).dropna().reset_index(drop=True)

# Step 4: Model Training & Evaluation
# Train XGBoost for amount and quantity
targets = ["amount", "quantity"]
models = {}
metrics = {}

for target in targets:
    X = featured_df.drop(columns=["month", "customer_id", "plant_id"] + targets)
    y = featured_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    metrics[target] = {"MAPE": mape, "RMSE": rmse}
    print(f"📊 {target}: MAPE={mape:.2f}, RMSE={rmse:.2f}")

# Plant-level validation
plant_test = featured_df.loc[X_test.index].copy()
plant_test['pred_amount'] = models['amount'].predict(X_test)
plant_test['pred_quantity'] = models['quantity'].predict(X_test)
plant_agg = plant_test.groupby('plant_id').agg({'amount': 'sum', 'pred_amount': 'sum', 'quantity': 'sum', 'pred_quantity': 'sum'})
plant_mape_amount = mean_absolute_percentage_error(plant_agg['amount'], plant_agg['pred_amount'])
plant_mape_quantity = mean_absolute_percentage_error(plant_agg['quantity'], plant_agg['pred_quantity'])
print(f"Plant-Level MAPE: Amount={plant_mape_amount:.2f}, Quantity={plant_mape_quantity:.2f}")
if plant_mape_amount <= 0.2 and plant_mape_quantity <= 0.2:
    print("Achieved >=80% accuracy at plant level")

# Step 5: Forecasting Next 12 Months
# Generate future features and predict
future_dfs = []
for (cust, plant), group in grouped:
    last_month = group["month"].max()
    future_dates = pd.date_range(forecast_start, periods=forecast_horizon, freq="M")
    future = pd.DataFrame({"month": future_dates, "customer_id": cust, "plant_id": plant})
    future = add_features(pd.concat([group.tail(12), future]))
    future = future[future["month"].isin(future_dates)]
    X_future = future.drop(columns=["month", "customer_id", "plant_id", "amount", "quantity"], errors="ignore")
    future["amount"] = models["amount"].predict(X_future)
    future["quantity"] = models["quantity"].predict(X_future)
    future_dfs.append(future)

forecasts = pd.concat(future_dfs)
forecasts.to_csv(os.path.join(output_dir, "forecasts.csv"), index=False)
print("✅ Forecasts saved to output/forecasts.csv")

# Step 6: Visualize Forecasts
# Plant-level forecast plot
plant_forecast = forecasts.groupby(["plant_id", "month"]).agg({"amount": "sum"})
plt.figure(figsize=(12, 6))
sns.lineplot(data=plant_forecast.reset_index(), x="month", y="amount", hue="plant_id")
plt.title("12-Month Plant-Level Forecasts (Oct 2025–Sep 2026)")
plt.savefig(os.path.join(output_dir, "plots", "forecast.png"))
plt.close()
```































