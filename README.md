
This project predicts hourly crowd levels (occupancy percentage) in a college gym using synthetic data stored in a SQLite database and traditional machine learning models. It demonstrates how to move from a basic model to a small, production‑like pipeline with SQL, Python scripts, model versioning, and a Streamlit dashboard.

## Model training

To train and save a model:

```bash
python train_model.py
## Data generation

To (re)create the SQLite database with synthetic gym data:

```bash
python data_generator.py

```bash
git add README.md
git commit -m "Add basic model training instructions to README"
git push

## Streamlit dashboard

To run the interactive dashboard:

```bash
streamlit run app.py


Then:

```bash
git add README.md
git commit -m "Document Streamlit app usage in README"
git push origin main
## Model history and versioning

Every time `train_model.py` is run, a new model is trained and saved into the `models/` folder, and a JSON line is appended to `model_history.json`. Each line looks like:

```json
{"model_name": "RandomForestRegressor", "rmse": 9.55, "mae": 7.67, "mape": 4140396.75, "timestamp": "20260222_192330", "model_path": "models/model_20260222_192330.pkl"}
