# Network Traffic Anomaly Detection Using Random Forest

A machine learning project that classifies network traffic as **Normal** or **Anomalous** using a Random Forest classifier, developed for CCE105 (University of Mindanao, College of Computing Education).

## Overview

Network traffic can show unusual patterns that indicate suspicious or abnormal activity. This project trains a supervised machine learning model to automatically classify traffic records as normal or anomalous based on four traffic measurements, instead of relying on manually defined rules.

## Dataset

- **Source:** [Network Anomaly Dataset on Kaggle](https://www.kaggle.com/datasets/amineipad/network-anoamly-dataset)
- **Records:** 1,654 network traffic records
- **Features:**
  - `Inbound Rate(bit/s)`
  - `Outbound Rate(bit/s)`
  - `Inbound Bandwidth Utilization(%)`
  - `Outbound Bandwidth Utilization(%)`
- **Target:** `Label` (0 = Normal, 1 = Anomaly)
- **Class balance:** 827 Normal / 827 Anomaly (perfectly balanced)

## Model

- **Algorithm:** Random Forest Classifier (scikit-learn), 100 estimators
- **Split:** 80% training / 20% testing, stratified to preserve class balance

## Results

| Metric | Value |
|---|---|
| Accuracy | 92.75% |
| Precision | 88.95% |
| Recall | 97.58% |
| F1-Score | 93.06% |

**Confusion Matrix:**

|  | Predicted: Normal | Predicted: Anomaly |
|---|---|---|
| **Actual: Normal** | 146 | 20 |
| **Actual: Anomaly** | 4 | 161 |

**Feature Importance:**
1. Inbound Bandwidth Utilization — 34.9%
2. Inbound Rate — 32.4%
3. Outbound Bandwidth Utilization — 17.1%
4. Outbound Rate — 15.6%

![Confusion Matrix](confusion_matrix.png)
![Feature Importance](feature_importance.png)

## How to Run

1. Clone this repository or download the files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the script:
   ```
   python network_traffic_anomaly_detection_using_random_forest.py
   ```

## Project Structure

```
├── network_traffic_anomaly_detection_using_random_forest.py   # Main script
├── networkanomalydataset_augmented.csv                        # Dataset
├── requirements.txt                                            # Python dependencies
├── confusion_matrix.png                                        # Output chart
├── feature_importance.png                                      # Output chart
├── label_distribution.png                                      # Output chart
└── README.md
```

## Authors

- [Student Name 1]
- [Student Name 2]

CCE105 — 1st Semester, 1st Term, S.Y. 2026–2027
University of Mindanao, College of Computing Education
