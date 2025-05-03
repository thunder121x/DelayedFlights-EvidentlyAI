import pandas as pd
import numpy as np
import os
from pathlib import Path
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import (
    MeanError, MAE, MAPE, RMSE, R2Score, AbsMaxError,
    DummyMAE, DummyMAPE, DummyRMSE
)

# Constants
CSV_PATH = "DelayedFlights.csv"
OUTPUT_PATH = "DelayedFlights_model_quality_report.html"
SAMPLE_SIZE = 5000
SEED_REFERENCE = 42
SEED_CURRENT = 43
FEATURES = ['DepTime', 'Distance', 'AirTime']
TARGET = 'ArrDelay'
PREDICTION = 'prediction'

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and drop rows with missing values."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def simulate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate prediction values by adding Gaussian noise."""
    df[PREDICTION] = df[TARGET] + np.random.normal(loc=0, scale=10, size=len(df))
    return df

def create_datasets(df: pd.DataFrame):
    """Sample reference and current datasets."""
    reference_data = df.sample(n=SAMPLE_SIZE, random_state=SEED_REFERENCE)
    current_data = df.sample(n=SAMPLE_SIZE, random_state=SEED_CURRENT)

    data_def = DataDefinition(
        regression=[Regression(target=TARGET, prediction=PREDICTION)]
    )

    reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_def)
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_def)
    
    return reference_dataset, current_dataset

def generate_report(reference_dataset: Dataset, current_dataset: Dataset, output_path: str):
    """Generate and save the model quality report."""
    report = Report(metrics=[
        MeanError(), MAE(), MAPE(), RMSE(), R2Score(), AbsMaxError(),
        DummyMAE(), DummyMAPE(), DummyRMSE()
    ])
    result = report.run(current_dataset, reference_dataset)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(output_path)
    print(f"✅ Model quality report saved to: {output_path}")
    return result

def main():
    try:
        df = load_and_clean_data(CSV_PATH)
        df = simulate_predictions(df)
        reference_dataset, current_dataset = create_datasets(df)
        result = generate_report(reference_dataset, current_dataset, OUTPUT_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()