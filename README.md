# ✈️ Airline Delay Model Quality Report with EvidentlyAI

## 📌 Project Overview

This project analyzes flight delay data using the **DelayedFlights.csv** dataset from Kaggle.  
We simulate predictions and evaluate model quality using **EvidentlyAI**, focusing on regression metrics.

## 📂 Dataset

- Source: [Kaggle - Airline Delay Causes](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses)
- Columns Used:
  - `DepTime` (Departure Time)
  - `Distance` (Flight Distance)
  - `AirTime` (Time in Air)
  - `ArrDelay` (Arrival Delay - Target)

## ⚙️ Workflow

1. Clean missing data
2. Simulate predictions by adding noise to `ArrDelay`
3. Split data into reference and current samples
4. Generate a model quality report using `evidently`

## 🧪 Output

- 📄 `DelayedFlights_model_quality_report.html`: A visual model performance report

## 💻 How to Run the Project

1. Clone the Repository

```bash
git clone https://github.com/thunder121x/DelayedFlights-EvidentlyAI.git
cd airline-delay-evidently
```

2. Dataset Download

Please manually download `DelayedFlights.csv` from [Kaggle](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses) and place it in the project root folder.

3. Prepare Environment (Mac & Windows)

✅ For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

✅ For Windows (CMD or PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

4. Open the Report

Once the script finishes, open DelayedFlights_model_quality_report.html in your browser to explore the results.

---

🧠 Key Insights

View the Evidently report to analyze:
- Prediction accuracy (MAE, RMSE, R²)
- How current data differs from reference
- Whether a new model is needed


📁 Files Included
- DelayedFlights.csv (dataset – must be downloaded separately)
- main.py (Python script for running analysis)
- requirements.txt
- DelayedFlights_model_quality_report.html (output report)
- README.md


🧑‍💻 Author

Nanphat Tongsirisukool


🙏 Acknowledgments
	•	Dataset: Airline Delay Causes – Kaggle
	•	Powered by EvidentlyAI

