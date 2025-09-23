# Portfolio Rebalancer ML API

A FastAPI-based project that uses **machine learning** to predict user risk category from their portfolio (equities, mutual funds, fixed deposits, bonds) and rebalance holdings to move more funds into **bonds** for better returns and reduced risk.

## 🚀 Features
- **Risk Scoring**: Predicts user’s risk category (Low / Medium / High) from holdings.
- **ML Models**: RandomForest Classifier (risk) + RandomForest Regressor (target bond %).
- **Portfolio Rebalancing**: Shifts funds from equities, mutual funds, and fixed deposits into bonds while respecting constraints.
- **API Endpoints**:
  - `POST /train` → Train models on synthetic dataset
  - `POST /predict_risk` → Predict risk category from user holdings
  - `POST /rebalance` → Suggest new allocations based on ML predictions
  - `GET /test` → Run built-in testcases

## 📂 Project Structure
```
portfolio_ml_api.py     # Main FastAPI app with ML logic
models/                 # Stores trained models (created after training)
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/portfolio-ml-api.git
cd portfolio-ml-api
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Linux / macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
pydantic
```

## ▶️ Running the API

Run with uvicorn:
```bash
python -m uvicorn portfolio_ml_api:app --reload
```

Open API docs in your browser:
👉 http://127.0.0.1:8000/docs

## 📊 Example Usage

### 1. Train Models
```bash
curl -X POST http://127.0.0.1:8000/train
```

### 2. Predict Risk
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"equity":50000,"mf":30000,"fd":10000,"bonds":10000}' \
http://127.0.0.1:8000/predict_risk
```

### 3. Rebalance Portfolio
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"equity":50000,"mf":30000,"fd":10000,"bonds":10000}' \
http://127.0.0.1:8000/rebalance
```

### 4. Run Testcases
```bash
curl http://127.0.0.1:8000/test
```

## 📌 Notes
- If models are missing, the app auto-trains them on startup.
- Synthetic data is used for demo purposes — in production you can replace it with real portfolio datasets.\

Test:
/predict_risk
i/p:
{
  "equity":100000 ,
  "mf": 2500000,
  "fd": 500000,
  "bonds": 150000
}
o/p:
{
  "risk_score": 1.7769230769230768,
  "risk_label": "High"
}

/rebalance
o/p:
{
  "original": {
    "equity": 100000,
    "mf": 2500000,
    "fd": 500000,
    "bonds": 150000
  },
  "suggested": {
    "equity": 100000,
    "mf": 1686549.09,
    "fd": 400000,
    "bonds": 1063450.91
  },
  "moves": {
    "from_equity": 0,
    "from_mf": 813450.91,
    "from_fd": 100000
  },
  "notes": "Target bonds pct = 0.327. Moved funds into bonds: total moved = 913450.91"
}
