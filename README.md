> 🔗 This is a forked project developed in collaboration with [@KoJenKang](https://github.com/KoJenKang) and [@Pin-Shiuan Liang](https://github.com/rachellg99).
# 🛒 Kaggle M5 Forecasting: Walmart Sales Prediction

This project implements a scalable deep learning solution for the [Kaggle M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) challenge. It predicts daily product demand using historical sales data, applying advanced **feature engineering** and a **Sequence-to-Sequence (Seq2Seq)** model.

---

## 📂 Project Structure

### `data engineering.ipynb`
- Performs **feature engineering** to extract time-series patterns:
  - Lag features (`lag_7`, `lag_28`)
  - Rolling means (`rolling_mean_7`, `rolling_mean_28`)
  - Calendar variables (`dayofweek`, `is_weekend`, etc.)
  - Event and price-based features
- Splits the dataset **by U.S. state** to enable memory-efficient training due to data volume.

### `sequence-to-sequence.ipynb`
- Builds a **Sequence-to-Sequence model with encoder-decoder architecture**
- Trains on each state’s data independently
- Generates **28-day ahead forecasts** using recursive decoding

### `lightGBM.ipynb`
- Builds a LightGBM model and generates 28-day ahead forecasts using all available historical data.
- Leverages past sales and model predictions to recursively forecast future demand.
---

## 🧠 Model: Sequence-to-Sequence (Seq2Seq)

A Seq2Seq model is a type of deep learning model that:
- Takes a **sequence of past sales** (e.g., 2 years)
- Outputs a **sequence of future sales** (e.g., next 28 days)

### How It Works
1. **Encoder** compresses historical sales into a context vector.
2. **Decoder** learns from that context to predict one day at a time.
3. **Recursive Prediction**: Each predicted day feeds into the next step to complete the 28-day forecast.

> 📊 Ideal for modeling **temporal dependencies** and generating multi-step forecasts in time-series data.

---

## 🌲 Model 2: LightGBM Baseline

**LightGBM** is a fast, tree-based machine learning algorithm ideal for tabular data.

### ✅ Key strengths:
- Builds **many small decision trees**
- Learns iteratively from residuals
- Optimized for **speed and large datasets**

Using a single LightGBM model provides a **strong baseline** to benchmark against deep learning approaches.

---
## 📌 Feature Engineering Summary

| Category              | Purpose                                                          | Example Features                        |
|-----------------------|------------------------------------------------------------------|-----------------------------------------|
| Lag features          | Capture seasonality and trends                                  | `lag_7`, `lag_28`                        |
| Rolling means         | Smooth out short-term noise                                     | `rolling_mean_7`, `rolling_mean_28`      |
| Calendar features     | Model recurring time-based patterns                             | `dayofweek`, `month`, `is_weekend`, etc.|
| Event flags           | Detect demand spikes from holidays/promotions                   | `is_event`                              |
| Price features        | Capture influence of price and discount changes                 | `price_change_rate`, `price_event_interaction` |
