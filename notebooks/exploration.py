"""
exploration.py
==============
Standalone Exploratory Data Analysis (EDA) script for Voyage Analytics.
Generates and saves visualisations to the notebooks/charts/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


def run_eda():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    charts_dir = project_root / "notebooks" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    users = pd.read_csv(data_dir / "users.csv")
    flights = pd.read_csv(data_dir / "flights.csv")
    hotels = pd.read_csv(data_dir / "hotels.csv")

    flights["date"] = pd.to_datetime(flights["date"])
    hotels["date"] = pd.to_datetime(hotels["date"])

    print("Generating Flights charts...")

    # 1. Price distribution
    plt.figure()
    sns.histplot(flights["price"], bins=30, kde=True, color="#00d4aa")
    plt.title("Flight Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(charts_dir / "flight_price_distribution.png", dpi=150)
    plt.close()

    # 2. Flight price by type
    plt.figure()
    sns.boxplot(x="flightType", y="price", data=flights, palette="viridis")
    plt.title("Flight Price by Category")
    plt.xlabel("Flight Category")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(charts_dir / "flight_price_by_type.png", dpi=150)
    plt.close()

    # 3. Flights booking trend
    plt.figure()
    flights_timeline = flights.groupby(flights["date"].dt.to_period("M")).size().reset_index(name="bookings")
    flights_timeline["date"] = flights_timeline["date"].dt.to_timestamp()
    sns.lineplot(x="date", y="bookings", data=flights_timeline, color="#f5a623", linewidth=2.5)
    plt.title("Monthly Flight Bookings Timeline")
    plt.xlabel("Date")
    plt.ylabel("Bookings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir / "flight_bookings_trend.png", dpi=150)
    plt.close()

    # 4. Distance vs price
    plt.figure()
    sns.scatterplot(x="distance", y="price", hue="flightType", data=flights.sample(1000, random_state=42), alpha=0.5)
    plt.title("Distance vs Price (Sampled)")
    plt.xlabel("Distance")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(charts_dir / "distance_vs_price.png", dpi=150)
    plt.close()

    print("Generating Hotels charts...")

    # 5. Hotel stays distribution
    plt.figure()
    sns.countplot(x="name", data=hotels, palette="coolwarm", order=hotels["name"].value_counts().index)
    plt.title("Hotel Bookings Count")
    plt.xlabel("Hotel Name")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(charts_dir / "hotel_popularity.png", dpi=150)
    plt.close()

    # 6. Hotel stay days
    plt.figure()
    sns.histplot(hotels["days"], bins=4, discrete=True, color="#ff6b6b")
    plt.title("Hotel Stay Duration (Days)")
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(charts_dir / "hotel_stay_days.png", dpi=150)
    plt.close()

    print("Generating Users charts...")

    # 7. Age distribution
    plt.figure()
    sns.histplot(users["age"], bins=20, kde=True, color="#8b5cf6")
    plt.title("User Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(charts_dir / "user_age_distribution.png", dpi=150)
    plt.close()

    # 8. Gender ratio
    plt.figure()
    users["gender"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#06b6d4", "#f472b6"], startangle=90)
    plt.ylabel("")
    plt.title("User Gender Split")
    plt.tight_layout()
    plt.savefig(charts_dir / "user_gender_ratio.png", dpi=150)
    plt.close()

    # 9. Correlation Analysis (Joined)
    print("Generating Joined charts...")
    flight_agg = flights.groupby("userCode").agg(
        total_flight_spend=("price", "sum"),
        flights_count=("travelCode", "count")
    ).reset_index()

    hotel_agg = hotels.groupby("userCode").agg(
        total_hotel_spend=("total", "sum"),
        hotels_count=("travelCode", "count")
    ).reset_index()

    merged = users.merge(flight_agg, left_on="code", right_on="userCode", how="left")
    merged = merged.merge(hotel_agg, on="userCode", how="left").fillna(0)
    merged["total_spend"] = merged["total_flight_spend"] + merged["total_hotel_spend"]

    plt.figure()
    sns.scatterplot(x="age", y="total_spend", hue="gender", data=merged, alpha=0.7)
    plt.title("User Age vs Total Travel Spend")
    plt.xlabel("Age")
    plt.ylabel("Total Spend ($)")
    plt.tight_layout()
    plt.savefig(charts_dir / "age_vs_spend.png", dpi=150)
    plt.close()

    print(f"EDA exploration completed. Saved 9 charts to {charts_dir}")


if __name__ == "__main__":
    run_eda()
