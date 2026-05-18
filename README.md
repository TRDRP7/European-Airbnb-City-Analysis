# 🏙️ European Airbnb City Analysis

A data analysis and machine learning project exploring Airbnb pricing patterns across major European cities — uncovering what drives prices, how cities compare, and building a predictive model for listing prices.

---

## 📌 Project Overview

This project performs end-to-end analysis on Airbnb listings from multiple European cities. It covers data aggregation, exploratory data analysis (EDA), visualization, and a Random Forest regression model to predict listing prices.

---

## 🌍 Cities Covered

Amsterdam, Athens, Barcelona, Berlin, Budapest, Lisbon, London, Paris, Rome, Vienna

---

## 📊 Analysis Performed

- **Price Distribution** — Overall spread of listing prices across all cities
- **Average Price by City** — Which cities are most and least expensive
- **Room Type Comparison** — How price varies by room type (entire home, private room, etc.)
- **Weekday vs Weekend Pricing** — Price differences between weekdays and weekends per city
- **Correlation Heatmap** — Relationships between price, distance, ratings, capacity, and more
- **Price Map** — Geographic scatter plot colored by price (latitude/longitude)
- **Superhost Impact** — Whether superhost status affects listing price
- **Guest Satisfaction vs Price** — Does higher satisfaction mean higher price?
- **Capacity & Bedrooms vs Price** — How size factors influence pricing

---

## 🤖 Machine Learning Model

A **Random Forest Regressor** was trained to predict listing prices based on key features:

| Feature | Description |
|---|---|
| `person_capacity` | Maximum number of guests |
| `cleanliness_rating` | Cleanliness score |
| `guest_satisfaction_overall` | Overall guest satisfaction |
| `bedrooms` | Number of bedrooms |
| `dist` | Distance to city center (km) |
| `metro_dist` | Distance to nearest metro (km) |
| `host_is_superhost` | Whether the host is a superhost |

**Metrics reported:** Mean Absolute Error (MAE) and R² Score

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading and manipulation |
| Matplotlib | Plotting and visualizations |
| Seaborn | Statistical visualizations |
| Scikit-learn | Machine learning model |
| OS | File handling |

---

## 📁 Project Structure

```
European-Airbnb-City-Analysis/
│
├── analysis.py                  # Main analysis script
│
├── *.csv                        # City datasets (weekday & weekend)
│
├── price_distribution.png       # Price distribution chart
├── avg_price_by_city.png        # Average price by city
├── price_by_room_type.png       # Room type comparison
├── weekday_vs_weekend.png       # Weekday vs weekend prices
├── correlation_heatmap.png      # Feature correlation heatmap
├── feature_importance.png       # ML model feature importance
├── dashboard.png                # Combined 6-chart dashboard
└── extended_dashboard.png       # Extended 9-chart dashboard
```

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/TRDRP7/European-Airbnb-City-Analysis.git
cd European-Airbnb-City-Analysis
```

2. **Install dependencies**
```bash
pip install pandas matplotlib seaborn scikit-learn
```

3. **Update the folder path** in `analysis.py` (line 7) to your local directory:
```python
folder = r'your/local/path/here'
```

4. **Run the script**
```bash
python analysis.py
```

All charts will be saved automatically in the project folder.

---

## 📈 Output Charts

| Chart | Description |
|---|---|
| `dashboard.png` | 6-panel overview of key insights |
| `extended_dashboard.png` | 9-panel deep dive including price map and ML results |

---

## 👤 Author

**Ram Prasaant**
GitHub: [@TRDRP7](https://github.com/TRDRP7)
