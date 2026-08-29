# Hotel Booking Cancellation Predictor

An end-to-end machine-learning application that predicts the probability that a hotel booking will be cancelled.

## 1. Problem Statement

Hotel booking cancellations create operational and revenue-management challenges. This project predicts cancellation risk from information known before the final booking outcome.

## 2. Dataset

Use the **Hotel Booking Demand** dataset that can be found on https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?resource=download and place:

`hotel_bookings.csv`

inside:

`data/`

The original dataset contains bookings for a resort hotel and a city hotel.



## 3. Project Structure

```text
hotel_booking_cancellation_app/
├── api.py
├── streamlit_app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── Dockerfile.api
├── sample_request.json
├── data/
│   └── hotel_bookings.csv
└── models/
    ├── hotel_cancellation_model.joblib
    └── metadata.json
```

## 4. Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```


## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Train the Model

```bash
python train.py
```

The script trains:
- Logistic Regression
- Random Forest
- Gradient Boosting

The model with the highest test F1 score is saved automatically as:

```text
models/hotel_cancellation_model.joblib
```

Evaluation information is written to:

```text
models/metadata.json
```

## 7. Run Streamlit Directly

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit application can make predictions directly using the local trained model.


## 8. Docker — Streamlit

First train the model so the model artifact exists:

```bash
python train.py
```

Build:

```bash
docker build -t hotel-cancellation-ui .
```

Run:

```bash
docker run -p 8501:8501 hotel-cancellation-ui
```


## 9. Cloud Deployment

A simple assignment deployment can use:

### Streamlit
Deploy the Streamlit Docker image to Azure Container Apps.


## 10. Architecture

```text
User
  |
  v
Streamlit Web UI
  |
  v
Scikit-learn Pipeline
  |-- preprocessing
  |-- trained classifier
  |
  v
Cancellation probability + risk level
```

## 11. ML Approach

The training pipeline:
1. Loads hotel booking data.
2. Selects predictors available before the booking outcome.
3. Handles missing numerical/categorical values.
4. Standardizes numerical predictors.
5. One-hot encodes categorical predictors.
6. Trains three classification algorithms.
7. Evaluates Accuracy, Precision, Recall, F1 and ROC-AUC.
8. Selects the best model by F1 score.
9. Saves the complete preprocessing + model pipeline.

## 12. Responsible AI Note

The model is an academic decision-support demonstration. It should not be the sole basis for decisions that negatively affect a customer. Performance, bias, drift, privacy and data quality should be monitored in a production implementation.
