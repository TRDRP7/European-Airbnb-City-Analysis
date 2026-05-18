import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. LOAD ALL CITIES ──────────────────────────────────────────────
folder = r'C:\Users\ram\Desktop\vault\python_2026\airbnb'

all_data = []
for file in os.listdir(folder):
    if file.endswith('.csv'):
        city = file.replace('.csv', '')
        temp = pd.read_csv(os.path.join(folder, file))
        temp['city'] = city
        all_data.append(temp)

df = pd.concat(all_data, ignore_index=True)
print(f"Total listings loaded: {len(df)}")
print(df['city'].value_counts())

# ── 2. PRICE DISTRIBUTION ───────────────────────────────────────────
plt.figure(figsize=(12, 5))
sns.histplot(df['realSum'], bins=80, color='steelblue')
plt.title('Price Distribution - All Cities')
plt.xlabel('Price (€)')
plt.ylabel('Number of listings')
plt.savefig(os.path.join(folder, 'price_distribution.png'))
plt.close()
print('Saved: price_distribution.png')

# ── 3. AVERAGE PRICE BY CITY ────────────────────────────────────────
city_prices = df.groupby('city')['realSum'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
city_prices.plot(kind='bar', color='coral')
plt.title('Average Price by City')
plt.xlabel('City')
plt.ylabel('Average Price (€)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'avg_price_by_city.png'))
plt.close()
print('Saved: avg_price_by_city.png')

# ── 4. ROOM TYPE COMPARISON ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='room_type', y='realSum')
plt.title('Price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price (€)')
plt.ylim(0, 2000)
plt.savefig(os.path.join(folder, 'price_by_room_type.png'))
plt.close()
print('Saved: price_by_room_type.png')

# ── 5. WEEKDAY vs WEEKEND ───────────────────────────────────────────
df['day_type'] = df['city'].apply(lambda x: 'Weekend' if 'weekend' in x else 'Weekday')
df['city_name'] = df['city'].str.replace('_weekdays', '').str.replace('_weekends', '')

weekend_vs_weekday = df.groupby(['city_name', 'day_type'])['realSum'].mean().unstack()
weekend_vs_weekday.plot(kind='bar', figsize=(14, 6), color=['steelblue', 'coral'])
plt.title('Weekday vs Weekend Prices by City')
plt.xlabel('City')
plt.ylabel('Average Price (€)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'weekday_vs_weekend.png'))
plt.close()
print('Saved: weekday_vs_weekend.png')

# ── 6. CORRELATION HEATMAP ──────────────────────────────────────────
num_cols = ['realSum', 'person_capacity', 'cleanliness_rating',
            'guest_satisfaction_overall', 'bedrooms', 'dist', 'metro_dist']
plt.figure(figsize=(10, 7))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'correlation_heatmap.png'))
plt.close()
print('Saved: correlation_heatmap.png')

# ── 7. PRICE PREDICTION MODEL ───────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

features = ['person_capacity', 'cleanliness_rating', 'guest_satisfaction_overall',
            'bedrooms', 'dist', 'metro_dist', 'host_is_superhost']
target = 'realSum'

model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nModel Results:")
print(f"MAE  : €{mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
importance.plot(kind='bar', color='teal')
plt.title('What Factors Affect Price the Most?')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'feature_importance.png'))
plt.close()
print('Saved: feature_importance.png')

print("\n✅ All done! Check your airbnb folder for the charts.")
# ── 8. DASHBOARD - ALL CHARTS IN ONE PAGE ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Airbnb European Cities - Full Analysis', fontsize=16, fontweight='bold')

# Chart 1 - Price Distribution
sns.histplot(df['realSum'], bins=80, color='steelblue', ax=axes[0,0])
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price (€)')

# Chart 2 - Avg Price by City
city_prices.plot(kind='bar', color='coral', ax=axes[0,1])
axes[0,1].set_title('Avg Price by City')
axes[0,1].tick_params(axis='x', rotation=45)

# Chart 3 - Room Type
sns.boxplot(data=df, x='room_type', y='realSum', ax=axes[0,2])
axes[0,2].set_title('Price by Room Type')
axes[0,2].set_ylim(0, 2000)

# Chart 4 - Weekday vs Weekend
weekend_vs_weekday.plot(kind='bar', color=['steelblue','coral'], ax=axes[1,0])
axes[1,0].set_title('Weekday vs Weekend')
axes[1,0].tick_params(axis='x', rotation=45)

# Chart 5 - Heatmap
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

# Chart 6 - Feature Importance
importance.plot(kind='bar', color='teal', ax=axes[1,2])
axes[1,2].set_title('Feature Importance')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(folder, 'dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: dashboard.png')
# ── 9. EXTRA CHARTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.suptitle('Airbnb - Extended Analysis', fontsize=18, fontweight='bold')

# Chart 1 - Superhost vs Non-superhost
sns.boxplot(data=df, x='host_is_superhost', y='realSum', ax=axes[0,0])
axes[0,0].set_title('Superhost vs Non-Superhost Prices')
axes[0,0].set_xlabel('Is Superhost?')
axes[0,0].set_ylabel('Price (€)')
axes[0,0].set_ylim(0, 2000)

# Chart 2 - Price vs Distance to city center
axes[0,1].scatter(df['dist'], df['realSum'], alpha=0.1, color='steelblue', s=5)
axes[0,1].set_title('Price vs Distance to City Center')
axes[0,1].set_xlabel('Distance (km)')
axes[0,1].set_ylabel('Price (€)')
axes[0,1].set_ylim(0, 3000)

# Chart 3 - Number of listings per city
listing_counts = df['city_name'].value_counts()
listing_counts.plot(kind='bar', color='mediumpurple', ax=axes[0,2])
axes[0,2].set_title('Number of Listings per City')
axes[0,2].set_xlabel('City')
axes[0,2].set_ylabel('Count')
axes[0,2].tick_params(axis='x', rotation=45)

# Chart 4 - Price vs Guest Satisfaction
axes[1,0].scatter(df['guest_satisfaction_overall'], df['realSum'], alpha=0.1, color='coral', s=5)
axes[1,0].set_title('Price vs Guest Satisfaction')
axes[1,0].set_xlabel('Guest Satisfaction Score')
axes[1,0].set_ylabel('Price (€)')
axes[1,0].set_ylim(0, 3000)

# Chart 5 - Capacity vs Price
sns.boxplot(data=df, x='person_capacity', y='realSum', ax=axes[1,1])
axes[1,1].set_title('Person Capacity vs Price')
axes[1,1].set_xlabel('Person Capacity')
axes[1,1].set_ylabel('Price (€)')
axes[1,1].set_ylim(0, 2000)

# Chart 6 - Top 10 most expensive cities
top10 = df.groupby('city')['realSum'].mean().sort_values(ascending=False).head(10)
top10.plot(kind='barh', color='tomato', ax=axes[1,2])
axes[1,2].set_title('Top 10 Most Expensive City/Day combos')
axes[1,2].set_xlabel('Average Price (€)')

# Chart 7 - Price map (lat/lng)
scatter = axes[2,0].scatter(df['lng'], df['lat'], c=df['realSum'],
                             cmap='YlOrRd', alpha=0.3, s=2, vmin=0, vmax=1000)
plt.colorbar(scatter, ax=axes[2,0])
axes[2,0].set_title('Price Map (Color = Price)')
axes[2,0].set_xlabel('Longitude')
axes[2,0].set_ylabel('Latitude')

# Chart 8 - Predicted vs Actual prices
axes[2,1].scatter(y_test, y_pred, alpha=0.2, color='teal', s=5)
axes[2,1].plot([0, 3000], [0, 3000], 'r--', linewidth=1)
axes[2,1].set_title('Predicted vs Actual Prices')
axes[2,1].set_xlabel('Actual Price (€)')
axes[2,1].set_ylabel('Predicted Price (€)')
axes[2,1].set_xlim(0, 3000)
axes[2,1].set_ylim(0, 3000)

# Chart 9 - Bedrooms vs Price
sns.boxplot(data=df, x='bedrooms', y='realSum', ax=axes[2,2])
axes[2,2].set_title('Bedrooms vs Price')
axes[2,2].set_xlabel('Number of Bedrooms')
axes[2,2].set_ylabel('Price (€)')
axes[2,2].set_ylim(0, 2000)

plt.tight_layout()
plt.savefig(os.path.join(folder, 'extended_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: extended_dashboard.png')