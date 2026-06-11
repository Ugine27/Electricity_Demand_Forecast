#  Electricity Demand Forecasting using LSTM

A beginner-friendly time-series forecasting project built with LSTM (Keras/TensorFlow).
Predicts next-day and next-week electricity demand using historical usage, temperature, and time features.

---

##  Project Preview

```
Raw Data → Preprocessing → Normalization → Sequences → LSTM Model → Predictions → Visualization
```

---

##  What is this project?

This project uses an **LSTM (Long Short-Term Memory)** neural network to forecast electricity demand.
LSTM is a type of RNN that can learn patterns in sequential/time-series data by "remembering" past values.

Given the past 24 hours of data (temperature, hour of day, day of week, demand), the model predicts the **next hour's electricity demand** — and by rolling this forward, it can forecast **24 hours or 7 days ahead**.

---

##  Project Structure

```
 electricity-demand-forecasting-lstm
 ┣  electricity_demand_forecasting_lstm.ipynb   ← Main Colab notebook
 ┗  README.md                                   ← You are here
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| TensorFlow / Keras | Building & training the LSTM model |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | MinMaxScaler for normalization |
| Matplotlib | Plotting actual vs predicted values |

>  All libraries come **pre-installed** in Google Colab — no setup needed.

---

##  Dataset

A **synthetic dataset** is generated inside the notebook simulating 2 years of realistic hourly electricity data:

| Feature | Description |
|---------|-------------|
| `datetime` | Hourly timestamps (2022–2023) |
| `temperature` | Simulated °C with seasonal + daily variation |
| `hour` | Hour of the day (0–23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `demand_mw` | Electricity demand in Megawatts |

**Patterns baked in:**
-  Daily peaks at morning & evening
-  Lower demand on weekends
-  Higher demand in summer (cooling) and winter (heating)
-  Random noise for realism

---

##  Model Architecture

```
Input: (24 time steps × 4 features)
        ↓
   LSTM(64 units)        ← learns sequential patterns
        ↓
   Dropout(0.2)          ← reduces overfitting
        ↓
   Dense(32, relu)       ← refines the output
        ↓
   Dense(1)              ← predicted demand value
```

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Early Stopping:** stops training if val_loss doesn't improve for 5 epochs

---

##  How to Run

### Option 1 — Google Colab (Recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `electricity_demand_forecasting_lstm.ipynb`
4. Click **Runtime → Run all** (`Ctrl + F9`)
5. Done — results appear in ~2–3 minutes 

### Option 2 — Local (Jupyter)
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter
jupyter notebook electricity_demand_forecasting_lstm.ipynb
```

---

##  Results

The notebook produces the following outputs:

-  **Training loss curve** — training vs validation loss per epoch
-  **Actual vs Predicted plot** — full 30-day test period + zoomed 7-day view
-  **Scatter plot** — predicted vs actual (closer to diagonal = better)
-  **24-hour forecast** — next day demand with recent context
-  **7-day forecast** — full next-week demand projection

**Evaluation Metrics:**
| Metric | Meaning |
|--------|---------|
| MAE | Average error in MW |
| RMSE | Penalizes large errors more |
| MAPE | Error as a percentage (easiest to interpret) |

---

##  Interview Cheatsheet

**Q: Why LSTM for time-series?**
> LSTM has a memory (cell state) that retains patterns over long sequences — unlike plain neural nets that forget earlier inputs.

**Q: Why normalize the data?**
> Neural networks are sensitive to scale. MinMaxScaler maps all features to [0, 1], making training faster and more stable.

**Q: What is a sliding window?**
> We use the past N steps as input to predict the next step. Sliding it forward creates many training samples from a single time series.

**Q: What does Dropout do?**
> Randomly ignores some neurons during training, forcing the model to generalize and reducing overfitting.

**Q: How would you improve this model?**
> Add more features (humidity, holidays), use a larger window, stack a second LSTM layer, or explore Transformer-based models.

---

##  Possible Improvements

- [ ] Use real-world dataset (e.g. UCI Electricity, Kaggle Energy datasets)
- [ ] Add holiday / weekend flags as features
- [ ] Try stacked LSTM (2 layers)
- [ ] Experiment with window sizes (48h, 72h)
- [ ] Compare with baseline models (Linear Regression, ARIMA)
- [ ] Add confidence intervals to forecasts

---

> *Clean LSTM pipeline: preprocessing → training → evaluation → forecasting → visualization*

---

##  License

This project is open-source and free to use for learning and portfolio purposes.
