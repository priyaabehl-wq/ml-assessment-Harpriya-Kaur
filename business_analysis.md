# Part B: Business Case Analysis

## B1. Problem Formulation

### (a) Machine Learning Problem Formulation

This problem can be formulated as a **supervised regression problem**.

- **Target Variable:**  
  `items_sold` (number of items sold per store per month)

- **Candidate Input Features:**
  - Store characteristics: store_size, location_type (urban/semi-urban/rural)
  - Promotion details: promotion_type (Flat Discount, BOGO, Free Gift, etc.)
  - Temporal factors: month, seasonality indicators
  - Customer behavior: footfall
  - Market conditions: competition_density
  - Demographics: customer profile (if available)

- **Type of ML Problem:**  
  Regression, because the target variable (`items_sold`) is continuous numeric.

**Justification:**  
The goal is to predict the quantity of items sold under different conditions. Since this is a numerical value, regression is the most appropriate approach. The model will learn relationships between promotions, store characteristics, and sales volume.

---

### (b) Why Items Sold is a Better Target than Revenue

Using **items sold (sales volume)** is more reliable than total revenue because:

- Revenue is influenced by **price variations**, discounts, and promotion types
- Different promotions (e.g., BOGO vs flat discount) affect pricing differently
- Revenue may increase even if fewer items are sold (due to higher prices)

In contrast:
- Items sold directly reflects **customer demand and engagement**
- It provides a clearer measure of **promotion effectiveness**

**Broader Principle:**
This illustrates that the target variable in machine learning should:
- Align closely with the **business objective**
- Be **stable and less affected by external distortions**
- Represent the **true outcome we want to optimize**

Choosing the wrong target can lead to misleading insights and poor decision-making.

---

### (c) Improved Modelling Strategy

Instead of using a single global model across all stores, a better approach is:

#### **Segmented or Hierarchical Modelling**

- Group stores based on characteristics such as:
  - Location type (urban, semi-urban, rural)
  - Store size
  - Customer demographics

- Train **separate models for each segment**, or use a **hierarchical (multi-level) model**

**Justification:**
- Different stores respond differently to the same promotion
- Urban stores may perform better with certain promotions (e.g., loyalty points)
- Rural stores may respond more to price-based offers (e.g., discounts)

This approach:
- Captures **local variations**
- Improves prediction accuracy
- Leads to more **personalized and effective promotion strategies**

---
## B2. Data and EDA Strategy

### (a) Data Integration and Dataset Design

The raw data consists of four tables:
- Transactions
- Store attributes
- Promotion details
- Calendar (weekend and festival flags)

#### Data Joining Strategy:
- **Transactions** is the primary table (fact table)
- Join **store attributes** using `store_id`
- Join **promotion details** using `promotion_type` or promotion identifier
- Join **calendar** using `transaction_date`

All joins would be performed using **left joins** to preserve all transaction records.

#### Grain of Final Dataset:
The modelling dataset should be aggregated to:
> **One row = one store per month**

This ensures consistency with the business goal of deciding monthly promotions per store.

#### Aggregations:
Before modelling, the following aggregations would be performed:

- Target:
  - Total `items_sold` per store per month

- Features:
  - Total or average `footfall`
  - Average `basket_size`
  - Promotion type used in that month
  - Count of promotional days
  - Proportion of weekend and festival days
  - Average competition density

This aggregation reduces noise and aligns the data with decision-making frequency.

---

### (b) Exploratory Data Analysis (EDA)

Before modelling, the following analyses would be performed:

#### 1. Target Distribution (Histogram of items_sold)
- **What to look for:** Skewness, outliers
- **Impact:** May require log transformation or outlier handling

#### 2. Promotion Type vs Items Sold (Boxplot / Bar Chart)
- **What to look for:** Which promotions drive higher sales
- **Impact:** Helps assess promotion effectiveness and guides feature importance

#### 3. Correlation Heatmap
- **What to look for:** Relationships between numerical features (e.g., footfall, competition density)
- **Impact:** Detect multicollinearity and select relevant features

#### 4. Time Series Trend (Items Sold over Time)
- **What to look for:** Seasonality, trends, spikes during festivals
- **Impact:** Helps engineer time-based features (month, season, festival flags)

#### 5. Location-wise Performance (Bar chart by location_type)
- **What to look for:** Differences across urban, semi-urban, rural stores
- **Impact:** Supports segmentation or hierarchical modelling

---

### (c) Impact of Promotion Imbalance

Since 80% of transactions occur without promotions, the dataset is highly imbalanced.

#### Potential Issues:
- The model may become biased toward predicting outcomes for non-promotion cases
- It may fail to learn the true impact of different promotion types

#### Mitigation Strategies:
- Ensure balanced representation using:
  - **Stratified sampling** at aggregation level
- Apply **feature engineering**:
  - Create a binary feature (promotion vs no promotion)
- Use **model evaluation carefully**:
  - Compare performance across promotion vs non-promotion subsets
- Consider **resampling techniques** or weighting methods

These steps ensure the model learns meaningful patterns from promotional data.

## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Evaluation Metrics

#### Train-Test Split Strategy:
Given that the data spans three years and is time-dependent, a **time-based split** should be used.

- Use the **first ~2.5 years (80%)** as the training set
- Use the **most recent ~0.5 years (20%)** as the test set

This ensures the model is trained on past data and evaluated on future data.

#### Why Random Split is Inappropriate:
- A random split would mix past and future data
- This leads to **data leakage**, where the model indirectly learns from future information
- It results in overly optimistic performance and poor real-world reliability

Time-based splitting ensures realistic evaluation aligned with how the model will be used in practice.

---

#### Evaluation Metrics:

1. **RMSE (Root Mean Squared Error)**
   - Measures the average magnitude of prediction errors
   - Penalizes large errors more heavily
   - **Interpretation:** Useful for identifying large mistakes in sales prediction, which could significantly impact inventory planning

2. **MAE (Mean Absolute Error)**
   - Measures the average absolute difference between predicted and actual values
   - Less sensitive to outliers than RMSE
   - **Interpretation:** Provides a clear understanding of average prediction error in terms of items sold

3. **Business Interpretation:**
   - Lower RMSE and MAE indicate better prediction accuracy
   - The company should prioritize models that minimize large forecasting errors (RMSE) while maintaining consistent accuracy (MAE)

---

### (b) Explaining Model Recommendations using Feature Importance

The model recommends different promotions for the same store in different months due to changes in input features.

#### Investigation Approach:

- Analyze **feature importance** from the trained model (e.g., Random Forest)
- Identify key features influencing predictions, such as:
  - Month (seasonality)
  - Festival or holiday indicators
  - Customer behavior patterns
  - Competition density

#### Example Explanation:

- In **December**, higher demand due to holidays and festivals may favor **Loyalty Points Bonus**, which encourages repeat purchases and customer retention
- In **March**, lower seasonal demand may require **Flat Discounts** to attract price-sensitive customers

#### Communication to Marketing Team:

- Use simple visualizations (e.g., feature importance charts)
- Explain that:
  - The model adapts recommendations based on **seasonal trends and customer behavior**
  - Different promotions work better under different conditions

This builds trust by showing that recommendations are **data-driven and context-specific**, not arbitrary.

---

### (c) Deployment and Monitoring Strategy

#### 1. Model Saving:
- Save the trained model using tools like `joblib` or `pickle`
- Store both the model and preprocessing pipeline together to ensure consistency

#### 2. Monthly Data Preparation:
- At the start of each month:
  - Collect updated data (store attributes, calendar info, competition data)
  - Apply the same preprocessing steps (feature engineering, encoding, scaling)
  - Ensure data format matches training data

#### 3. Prediction Pipeline:
- Load the saved model
- Input the prepared data for all 50 stores
- Generate predictions for each promotion scenario
- Select the promotion that maximizes predicted items_sold for each store

#### 4. Monitoring and Maintenance:

To ensure long-term reliability:

- **Performance Monitoring:**
  - Track RMSE and MAE over time
  - Compare predicted vs actual sales monthly

- **Data Drift Detection:**
  - Monitor changes in feature distributions (e.g., customer behavior shifts)

- **Concept Drift Detection:**
  - Check if relationships between features and sales are changing

- **Retraining Trigger:**
  - Retrain the model when performance degrades beyond a threshold
  - Example: Significant increase in RMSE over consecutive months

#### Final Outcome:
This pipeline ensures that the model delivers consistent, up-to-date, and reliable promotion recommendations for all stores.