# Hotel Booking Cancellation Predictor

An end-to-end AI/ML web application that predicts the probability that a hotel reservation will be cancelled. The solution covers data preparation, machine-learning model training and evaluation, a browser-based prediction interface, and deployment to Google Cloud Run.

---

## 1. Problem Statement

Hotel booking cancellations create operational and revenue-management challenges for hotels. A cancelled reservation can lead to lost room revenue, reduced occupancy, inefficient room allocation, additional administrative work, and uncertainty in forecasting.

The objective of this project is to develop an AI-based system that estimates cancellation risk using reservation information available before the final booking outcome is known.

The system is intended to support hotel staff by identifying bookings that may require additional monitoring or customer confirmation.

---

## 2. Use Case

The application can be used by hotel reservation teams, revenue-management teams, and operational staff to assess the cancellation risk of individual bookings.

Users enter booking information such as:

- Hotel type
- Lead time before arrival
- Number of nights
- Number of adults and children
- Deposit type
- Customer type
- Previous cancellations
- Average daily rate
- Special requests
- Meal plan
- Market segment
- Repeated-guest status
- Parking requirement

The application then provides:

- Predicted cancellation probability
- Risk classification
- Recommended operational action

Typical uses include identifying high-risk reservations, prioritising bookings for customer confirmation, supporting occupancy planning, and assisting reservation-management decisions.

The application is a decision-support tool and should not be used as the sole basis for customer-impacting decisions.

---

## 3. Solution Overview

The solution follows an end-to-end machine-learning lifecycle:

```text
Hotel Booking Dataset
        |
        v
Data Cleaning
        |
        v
Feature Selection / Engineering
        |
        v
Preprocessing
        |
        v
Train Multiple ML Models
        |
        v
Model Evaluation
        |
        v
Select Best Model
        |
        v
Save Trained Pipeline
        |
        v
Web Application
        |
        v
Google Cloud Deployment
        |
        v
Cancellation Probability + Risk Level
```

The training process evaluates multiple classification algorithms and automatically selects the best-performing model using the F1 score.

The complete preprocessing and prediction pipeline is saved as a model artifact. The web application loads this model and generates predictions from reservation data entered by the user.

---

## 4. Dataset

This project uses the **Hotel Booking Demand** dataset.

**Kaggle:**  
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The dataset is based on the Hotel Booking Demand dataset published by Nuno António, Ana de Almeida, and Luís Nunes in *Data in Brief* (2019).

The dataset contains approximately **119,000 hotel bookings** from:

- A resort hotel
- A city hotel

The target variable is:

```text
is_canceled
```

where:

```text
0 = Booking was not cancelled
1 = Booking was cancelled
```

Download `hotel_bookings.csv` and place it inside:

```text
data/
```

Example:

```text
hotel_booking_cancellation_app/
└── data/
    └── hotel_bookings.csv
```

### Data Leakage Prevention

Fields that reveal the final reservation outcome are excluded from the prediction features.

Examples include:

```text
reservation_status
reservation_status_date
```

These variables could reveal the final booking outcome and therefore cause data leakage.

---

## 5. AI/ML Approach

The project treats booking cancellation as a **binary classification problem**.

### Models Evaluated

The training script evaluates:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Preprocessing

Numerical features are processed using:

```text
Missing Value Handling
        |
        v
Median Imputation
        |
        v
Standard Scaling
```

Categorical features are processed using:

```text
Missing Value Handling
        |
        v
Most-Frequent Imputation
        |
        v
One-Hot Encoding
```

The preprocessing steps and classifier are combined using a Scikit-learn pipeline.

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The model with the highest F1 score is selected as the production model.

The trained model pipeline is saved as:

```text
models/hotel_cancellation_model.joblib
```

Evaluation metadata is saved as:

```text
models/metadata.json
```

### Risk Classification

The web application converts the predicted cancellation probability into three operational risk categories:

| Probability | Risk Level |
|---|---|
| Below 40% | Low Risk |
| 40% – 69% | Moderate Risk |
| 70% and above | High Risk |

These thresholds are used for user-friendly interpretation of the model output.

---

## 6. Application Architecture

### Application Flow

```text
                  ┌─────────────────────────────┐
                  │            User             │
                  │          Web Browser        │
                  └──────────────┬──────────────┘
                                 |
                                 v
                  ┌─────────────────────────────┐
                  │      StaySure Web App       │
                  │      Reservation Form       │
                  └──────────────┬──────────────┘
                                 |
                                 v
                  ┌─────────────────────────────┐
                  │   Scikit-learn Pipeline     │
                  │                             │
                  │  Numerical Preprocessing    │
                  │  Categorical Preprocessing  │
                  │  Trained Classifier         │
                  └──────────────┬──────────────┘
                                 |
                                 v
                  ┌─────────────────────────────┐
                  │ Cancellation Probability    │
                  │ Risk Classification         │
                  │ Recommended Action          │
                  └─────────────────────────────┘
```

### Cloud Deployment Architecture

```text
GitHub Repository
        |
        v
Google Cloud Build
        |
        v
Google Buildpacks
        |
        v
Container Image
        |
        v
Artifact Registry
        |
        v
Google Cloud Run
        |
        v
Public Web Application
```

This deployment approach allows the application to be built and deployed in the cloud without requiring a locally built Docker image.

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.14 |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Source Control | Git / GitHub |
| Cloud Platform | Google Cloud Platform |
| Build Service | Google Cloud Build |
| Build Method | Google Buildpacks |
| Container Registry | Google Artifact Registry |
| Cloud Hosting | Google Cloud Run |
| Optional Containerisation | Docker |

---

## 8. Project Structure

```text
hotel_booking_cancellation_app/
│
├── streamlit_app.py
├── train.py
├── requirements.txt
├── .python-version
├── README.md
├── Dockerfile
│
├── data/
│   └── hotel_bookings.csv
│
└── models/
    ├── hotel_cancellation_model.joblib
    └── metadata.json
```

Additional files may be present depending on the development version of the project.

---

## 9. Local Setup Instructions

### Prerequisites

Install:

- Python 3.14
- Git
- pip

Verify Python:

```bash
python --version
```

Expected:

```text
Python 3.14.x
```

### Clone the Repository

```bash
git clone https://github.com/dhkdinuka/hotel_booking_cancellation_app.git
cd hotel_booking_cancellation_app
```

### Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 10. Train the Model

Place the dataset at:

```text
data/hotel_bookings.csv
```

Then run:

```bash
python train.py
```

The training script:

1. Loads the hotel booking dataset
2. Selects prediction features
3. Handles missing values
4. Preprocesses numerical and categorical variables
5. Trains Logistic Regression
6. Trains Random Forest
7. Trains Gradient Boosting
8. Evaluates model performance
9. Selects the best model using F1 score
10. Saves the complete prediction pipeline
11. Saves model evaluation metadata

The resulting model is stored at:

```text
models/hotel_cancellation_model.joblib
```

Metadata is stored at:

```text
models/metadata.json
```

---

## 11. Run the Web Application Locally

After installing dependencies and ensuring the trained model exists, run:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The application will load the saved model and provide an interactive hotel booking cancellation-risk interface.

---

## 12. Deployment Details

The application is deployed using **Google Cloud Run**.

### Public Deployment URL

The live application is publicly accessible at:

https://hotel-booking-cancellation-app-139031721804.asia-south1.run.app/

### Source Code Repository

The complete project source code is available on GitHub:

https://github.com/dhkdinuka/hotel_booking_cancellation_app

### Deployment Flow

```text
GitHub Repository
        |
        v
Google Cloud Build
        |
        v
Google Buildpacks
        |
        v
Artifact Registry
        |
        v
Google Cloud Run
```

### Deployment Process

1. The source code is stored in GitHub.
2. Google Cloud Build retrieves the application source.
3. Google Buildpacks detect the Python application.
4. Dependencies are installed from `requirements.txt`.
5. A deployable container image is created.
6. The image is stored in Artifact Registry.
7. Cloud Run creates a service revision.
8. The application is exposed through a public HTTPS endpoint.

### Python Runtime

The repository contains:

```text
.python-version
```

with:

```text
3.14
```

### Application Entrypoint

The Cloud Run deployment starts the application using:

```bash
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

`$PORT` is supplied automatically by Cloud Run.

### Cloud Run Configuration

```text
Platform: Google Cloud Run
Region: asia-south1
CPU: 1
Memory: 1 GiB
Minimum Instances: 0
Maximum Instances: 1
Authentication: Public access
```

The application uses 1 GiB memory because the trained ML pipeline requires more memory than the smallest Cloud Run memory configuration.

### Continuous Deployment

```text
Code Change
    |
    v
Git Commit
    |
    v
Push to GitHub
    |
    v
Cloud Build Trigger
    |
    v
Build Application
    |
    v
Deploy New Cloud Run Revision
```

---

## 13. API/Web Application Usage

This project is primarily delivered as a **web application**.

### Step 1 – Open the Application

Open the public Google Cloud Run URL in a web browser.

### Step 2 – Enter Booking Information

Provide the main reservation attributes:

- Hotel type
- Lead time
- Number of nights
- Average daily rate
- Adults
- Children
- Deposit type
- Customer type
- Previous cancellations
- Special requests

Additional reservation attributes include:

- Meal plan
- Market segment
- Repeated-guest status
- Parking requirement

### Step 3 – Run the Prediction

Select:

```text
Run Cancellation Risk Assessment
```

### Step 4 – Review the Result

The application displays:

```text
Cancellation Probability
        +
Risk Classification
        +
Recommended Action
```

Example:

```text
Cancellation Probability: 76.4%

HIGH RISK

Recommended Action:
Prioritise the reservation for confirmation and consider
contacting the customer before arrival.
```

### Step 5 – Review Additional Details

Users can scroll down to review:

- Risk classification thresholds
- Operational recommendation
- Booking summary

---

## 14. Docker Instructions

Docker is **not required for the current production deployment** because Google Cloud Buildpacks automatically create the container image used by Google Cloud Run.

However, Docker support can be used for portability or local container testing.

### Train the Model First

Ensure the trained model exists:

```text
models/hotel_cancellation_model.joblib
```

If required:

```bash
python train.py
```

### Build the Docker Image

```bash
docker build -t hotel-cancellation-app .
```

### Run the Container

```bash
docker run -p 8501:8501 hotel-cancellation-app
```

Open:

```text
http://localhost:8501
```

### Example Dockerfile

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
```

For the submitted cloud implementation, the primary deployment method is **Cloud Build + Google Buildpacks + Cloud Run**.

---

## 15. Model Input Features

The deployed prediction application uses the following features:

```text
hotel
lead_time
stays_in_weekend_nights
stays_in_week_nights
adults
children
babies
meal
market_segment
distribution_channel
is_repeated_guest
previous_cancellations
previous_bookings_not_canceled
reserved_room_type
deposit_type
customer_type
adr
required_car_parking_spaces
total_of_special_requests
```

Some values that are not entered directly by the user are derived or assigned by the web application before prediction.

The same preprocessing logic used during training is stored inside the Scikit-learn model pipeline, ensuring consistent preprocessing during inference.

---


## 16. Future Improvements

Possible future improvements include:

- Explainable AI using SHAP or feature importance
- User authentication
- Reservation-system integration

---

## 17. Summary

This project demonstrates a complete cloud AI lifecycle:

```text
Real-World Problem
        |
        v
Dataset
        |
        v
Data Preprocessing
        |
        v
Machine Learning
        |
        v
Model Evaluation
        |
        v
Model Persistence
        |
        v
Web Application
        |
        v
Source Control
        |
        v
Cloud Build
        |
        v
Cloud Deployment
        |
        v
Public AI Application
```

The resulting system provides a practical browser-based interface for estimating hotel booking cancellation risk and demonstrates how an AI model can be trained, packaged, deployed, and consumed as a cloud-hosted application.
