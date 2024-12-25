import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = quote_plus(os.getenv("DATABASE_PASSWORD"))
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Connect to the database using SQLAlchemy
DATABASE_URI = f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}"
engine = create_engine(DATABASE_URI, echo=True)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Function to fetch joined data from MedCheckUp, ResultMedCheckUp, and Child
def fetch_joined_data(start_year, end_year):
    query = text(f"""
    SELECT
        MedCheckUp.id AS med_check_up_id,
        MedCheckUp.child_id,
        Child.gender,
        MedCheckUp.height,
        MedCheckUp.weight,
        MedCheckUp.age,
        MedCheckUp.circumference,
        MedCheckUp.created_at AS med_check_up_created_at,
        MedCheckUp.updated_at AS med_check_up_updated_at,
        ResultMedCheckUp.imt,
        ResultMedCheckUp.ipb,
        ResultMedCheckUp.status
    FROM
        MedCheckUp
        LEFT JOIN ResultMedCheckUp ON MedCheckUp.id = ResultMedCheckUp.med_check_up_id
        LEFT JOIN Child ON MedCheckUp.child_id = Child.id
    WHERE
        YEAR(MedCheckUp.updated_at) BETWEEN :start_year AND :end_year
    """)
    result = session.execute(query, {'start_year': start_year, 'end_year': end_year})
    rows = result.fetchall()
    
    if rows:
        # Extract column names from the result
        columns = result.keys()
        # Convert rows to dictionaries using column names
        data = [dict(zip(columns, row)) for row in rows]
        # Create DataFrame from the list of dictionaries
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

# Function to filter data by year range
def filter_data_by_year(df, start_year, end_year):
    return df[(df["med_check_up_updated_at"].dt.year >= start_year) & (df["med_check_up_updated_at"].dt.year <= end_year)]

# Function to train model and make predictions
def train_and_predict(start_year, end_year):
    df = fetch_joined_data(start_year, end_year)

    # Convert 'med_check_up_updated_at' to datetime
    df["med_check_up_updated_at"] = pd.to_datetime(df["med_check_up_updated_at"])

    df_filtered = filter_data_by_year(df, start_year, end_year)

    # Encoding categorical variables
    df_filtered.loc[:, "gender"] = df_filtered["gender"].map({"male": 0, "female": 1})
    df_filtered.loc[:, "status"] = df_filtered["status"].map({"normal": 0, "stunting": 1, "overweight": 2, "obese": 3})

    # Filter only data with "stunting" status
    df_stunting = df_filtered[df_filtered["status"] == 1]

    # Check if df_stunting is not empty
    if not df_stunting.empty:
        # Features and target
        X = df_stunting[["age", "height", "weight", "circumference", "imt", "ipb"]]
        y = df_stunting["gender"]  # Target: gender (0 for male, 1 for female)

        # Data for regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Regression model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = accuracy_score(y_test.round().astype(int), [round(pred) for pred in y_pred])

        # Hitung nilai rata-rata untuk setiap fitur berdasarkan gender
        avg_values_male = df_stunting[df_stunting["gender"] == 0].mean()
        avg_values_female = df_stunting[df_stunting["gender"] == 1].mean()

        # DataFrame untuk proyeksi
        avg_male_df = pd.DataFrame([avg_values_male[["age", "height", "weight", "circumference", "imt", "ipb"]]]).rename(columns=str)
        avg_female_df = pd.DataFrame([avg_values_female[["age", "height", "weight", "circumference", "imt", "ipb"]]]).rename(columns=str)

        # Proyeksi jumlah kasus stunting di setiap bulan selama 12 bulan ke depan
        future_predictions = []
        pred_year = end_year + 1
        for i in range(12):
            # Mengatur tanggal prediksi
            prediction_date = datetime(pred_year, 1, 1) + timedelta(days=30 * i)
            prediction_date_str = prediction_date.strftime("%Y-%m")
            
            # Update 'days_since_epoch' untuk setiap bulan
            avg_male_df["days_since_epoch"] = (avg_male_df.index + 1) * 30
            avg_female_df["days_since_epoch"] = (avg_female_df.index + 1) * 30
            
            # Menggunakan rata-rata jumlah prediksi untuk bulan tersebut
            projected_male = model.predict(avg_male_df)[0] + np.random.uniform(-5, 5)
            projected_female = model.predict(avg_female_df)[0] + np.random.uniform(-5, 5)
            
            future_predictions.append({
                'month': prediction_date_str,
                'predicted_male': max(0, round(projected_male, 3)),
                'num_pred_male': max(0, round(projected_male)),
                'predicted_female': max(0, round(projected_female, 3)),
                'num_pred_female': max(0, round(projected_female))
            })

        # Hitung total dan rata-rata untuk setahun
        total_pred_male = sum(pred['num_pred_male'] for pred in future_predictions)
        total_pred_female = sum(pred['num_pred_female'] for pred in future_predictions)
        avg_prob_male = sum(pred['predicted_male'] for pred in future_predictions) / 12
        avg_prob_female = sum(pred['predicted_female'] for pred in future_predictions) / 12

        # Compile results
        result = {
            'code': 200,
            'message': 'Data berhasil dikembalikan',
            'data': {
                'prediksi': [
                    {'Tahun Prediksi': pred_year},
                    {'name': 'Laki - Laki', 'data': [pred['num_pred_male'] for pred in future_predictions]},
                    {'name': 'Perempuan', 'data': [pred['num_pred_female'] for pred in future_predictions]}
                ],
                'categories': ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
            }
        }

        return result
    else:
        return None
