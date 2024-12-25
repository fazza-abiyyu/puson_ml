import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
from datetime import datetime
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

# Set default years
current_year = datetime.now().year
default_start_year = current_year - 1
default_end_year = current_year

# Function to fetch joined data from MedCheckUp, ResultMedCheckUp, and Child
def fetch_joined_data(start_year=default_start_year, end_year=default_end_year):
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

# Fetch joined data and create DataFrame
df = fetch_joined_data()

# Close the session
session.close()

# Convert 'med_check_up_updated_at' to datetime
df["med_check_up_updated_at"] = pd.to_datetime(df["med_check_up_updated_at"])

# Function to filter data by year range
def filter_data_by_year(df, start_year, end_year):
    return df[(df["med_check_up_updated_at"].dt.year >= start_year) & (df["med_check_up_updated_at"].dt.year <= end_year)]

# Function to train model and make predictions
def train_and_predict(start_year=default_start_year, end_year=default_end_year):
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
        
        # Convert regression predictions to binary classes
        y_test_rounded = y_test.round().astype(int)
        y_pred_rounded = [round(pred) for pred in y_pred]
        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)

        # Calculate average values for each feature based on gender
        avg_values_male = df_stunting[df_stunting["gender"] == 0].mean()
        avg_values_female = df_stunting[df_stunting["gender"] == 1].mean()

        # DataFrame for projections
        avg_male_df = pd.DataFrame([avg_values_male[["age", "height", "weight", "circumference", "imt", "ipb"]]]).rename(columns=str)
        avg_female_df = pd.DataFrame([avg_values_female[["age", "height", "weight", "circumference", "imt", "ipb"]]]).rename(columns=str)

        # Project future stunting cases using average values
        future_cases_male = model.predict(avg_male_df)
        future_cases_female = model.predict(avg_female_df)

        # Number of predictions based on gender
        num_pred_male = sum([1 for pred in y_pred_rounded if pred == 0])
        num_pred_female = sum([1 for pred in y_pred_rounded if pred == 1])
        
        return accuracy, mae, r2, future_cases_male, future_cases_female, num_pred_male, num_pred_female
    else:
        return 0, 0, 0, [0], [0], 0, 0
