
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

