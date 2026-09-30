# Sequence-Based Deep Learning System for Network Intrusion Detection and Threat Classification Using LSTM and GRU

> A deep learning–based network intrusion detection system that learns patterns from sequential network-flow data to identify malicious traffic and classify network threats using **LSTM and GRU architectures**.

---


## Overview

Modern networks generate large volumes of traffic that must be continuously monitored for malicious activity. Traditional rule-based intrusion detection systems can struggle with evolving attack patterns and large-scale network traffic.

This project develops a **sequence-based deep learning system for Network Intrusion Detection (NIDS)** using the **CIC-IDS2017** dataset.

The system processes network-flow features, performs leakage-aware preprocessing, constructs flow sequences, and trains recurrent neural networks to distinguish between **benign and malicious network traffic**.

The primary deep learning models investigated are:

* **LSTM — Long Short-Term Memory**
* **GRU — Gated Recurrent Unit**
![Image Alt Text]([assets/my-image.png](https://github.com/KatparSarfraz1214/Sequence-Based-Deep-Learning-System-for-Network-Intrusion-Detection-and-Threat-Classification/blob/main/Screenshot%202026-09-22%20194740.png?raw=true))


The project focuses on building a complete machine-learning pipeline rather than only training a neural network:

```text
Raw Network Flows
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train/Validation/Test Strategy
       ↓
Feature Scaling
       ↓
Sequence Construction
       ↓
LSTM / GRU
       ↓
Threat Prediction
       ↓
Evaluation
       ↓
Deployment-Ready Model
```

---

## Key Features

* Network intrusion detection using deep learning
* Sequence-based modeling of network-flow data
* LSTM and GRU architectures
* Binary benign-vs-attack detection
* Support for multiclass threat classification
* Handling of missing and infinite values
* Duplicate-flow removal
* Training-only feature scaling
* Class-imbalance handling
* Early stopping and learning-rate scheduling
* Precision, Recall, F1-score and ROC-AUC evaluation
* Confusion-matrix analysis
* Day-based generalization testing
* Saved model and preprocessing artifacts
* Deployment-ready inference pipeline

---

# Problem Statement

Network intrusion detection requires identifying malicious traffic among a large volume of legitimate network activity.

The objective of this project is to develop a deep learning system capable of learning patterns from network-flow sequences and detecting malicious activity.

### Primary objective

Given a sequence of network-flow feature vectors:

```text
X = [x₁, x₂, x₃, ..., xₜ]
```

predict whether the observed traffic represents:

```text
0 → BENIGN
1 → ATTACK
```

### Extended objective

The system can be extended to classify specific attack categories such as:

* DDoS
* DoS
* Port Scanning
* Brute Force
* Botnet
* Web Attacks
* Infiltration

---

# Dataset

## CIC-IDS2017

The project uses the **CIC-IDS2017** intrusion-detection benchmark from the Canadian Institute for Cybersecurity.

The dataset contains labeled network flows with more than 80 traffic-related features and includes benign traffic together with multiple attack scenarios.

Dataset:

**CIC-IDS2017 — Canadian Institute for Cybersecurity**

Official source:

https://www.unb.ca/cic/datasets/ids-2017.html

### Dataset files used

The project uses the `MachineLearningCSV.zip` distribution containing:

```text
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
Friday-WorkingHours-Morning.pcap_ISCX
Monday-WorkingHours.pcap_ISCX
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX
Tuesday-WorkingHours.pcap_ISCX
Wednesday-workingHours.pcap_ISCX
```

---

# Attack Scenarios

The dataset contains traffic associated with multiple network-security scenarios, including:

| Category     | Examples                      |
| ------------ | ----------------------------- |
| Benign       | Normal network traffic        |
| DoS          | Denial-of-Service attacks     |
| DDoS         | Distributed Denial-of-Service |
| Port Scan    | Network reconnaissance        |
| Brute Force  | FTP/SSH brute-force traffic   |
| Web Attacks  | Application-layer attacks     |
| Botnet       | Botnet traffic                |
| Infiltration | Network infiltration          |

The initial modeling task uses:

```text
BENIGN
ATTACK
```

as the binary classification target.

The architecture can subsequently be adapted for multiclass threat classification.

---

# Why LSTM and GRU?

Traditional machine-learning models generally process each flow independently.

This project investigates recurrent architectures because network traffic can be represented as an ordered sequence of observations.

A sequence can be represented as:

```text
Flow₁ → Flow₂ → Flow₃ → ... → Flowₜ
```

The model learns representations from these sequences before producing a threat prediction.

## LSTM

Long Short-Term Memory networks use gated mechanisms to retain and discard information across sequence steps.

The core gates are:

* Forget gate
* Input gate
* Candidate state
* Output gate

LSTM is useful when dependencies across multiple sequence steps need to be modeled.

## GRU

Gated Recurrent Units provide a simpler recurrent architecture using:

* Update gate
* Reset gate

GRUs generally have fewer parameters than LSTMs while still providing mechanisms for learning sequential dependencies.

The project compares both architectures rather than assuming one is superior.

---

# System Architecture

```text
                  CIC-IDS2017
                       │
                       ▼
              Dataset Ingestion
                       │
                       ▼
               Data Validation
                       │
                       ▼
              Cleaning & Deduplication
                       │
                       ▼
              Missing/Inf Handling
                       │
                       ▼
             Feature Preparation
                       │
                       ▼
          Train / Validation / Test
                       │
                       ▼
           StandardScaler
          (TRAIN DATA ONLY)
                       │
                       ▼
            Sequence Construction
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
           LSTM                 GRU
             │                   │
             └─────────┬─────────┘
                       ▼
                Threat Prediction
                       │
                       ▼
              Model Evaluation
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Precision   Recall      F1
                       │
                       ▼
                  Deployment
```

---

# Data Preprocessing

The preprocessing pipeline is designed to avoid common problems encountered when working with CIC-IDS2017.

## 1. Column normalization

Feature names are cleaned to provide consistent identifiers.

```python
df.columns = (
    df.columns
    .str.strip()
)
```

---

## 2. Duplicate removal

Duplicate network-flow records are removed before modeling.

```python
df = df.drop_duplicates()
```

This reduces unnecessary repetition in the training data.

---

## 3. Infinity handling

Some network-flow calculations can produce infinite values.

```python
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)
```

---

## 4. Missing-value handling

Missing numerical values are handled using statistics calculated from the training data.

This prevents validation/test information from influencing preprocessing.

---

## 5. Feature selection

Non-numeric and constant features are removed.

The resulting input consists of numerical network-flow features suitable for neural-network processing.

---

# Data Splitting Strategy

A major focus of the project is **generalization rather than maximizing an artificially high accuracy score**.

Instead of randomly distributing individual flows across training and test sets, the project uses a **day-based holdout strategy**.

### Training

```text
Monday
Tuesday
Wednesday
Thursday Web Attacks
```

### Validation

```text
Thursday Infiltration
```

### Final Test

```text
Friday Morning
Friday DDoS
Friday PortScan
```

This provides a more demanding evaluation because the final test traffic comes from held-out dataset files.

The test set remains untouched during model development and hyperparameter tuning.

---

# Preventing Data Leakage

Data leakage can produce overly optimistic machine-learning results.

This project therefore follows several precautions.

### Scaling

The scaler is fitted only on training data:

```python
scaler.fit(X_train)
```

and then applied to validation/test data:

```python
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

### Imputation

Missing-value statistics are derived from training data and reused for validation/test data.

### Test isolation

The final Friday test data is not used for:

* Model training
* Hyperparameter selection
* Early stopping decisions
* Threshold optimization

This allows the final test set to provide an independent estimate of model generalization.

---

# Sequence Construction

The recurrent models require 3-dimensional input:

```text
(samples, timesteps, features)
```

For example:

```text
(samples, 10, 78)
```

where:

* `samples` = number of sequences
* `10` = sequence length
* `78` = network-flow features

A sequence can be represented as:

```text
          Feature Vector
                │
Flow 1 ─────────┤
Flow 2 ─────────┤
Flow 3 ─────────┤
...             ├──→ LSTM / GRU
Flow 10 ────────┤
                │
                ▼
        BENIGN / ATTACK
```

Sequences are constructed separately within each dataset file to prevent sequences from crossing dataset boundaries.

---

# Model Architecture

## LSTM

The primary architecture consists of:

```text
Input
  ↓
LSTM(128)
  ↓
Dropout(0.3)
  ↓
LSTM(64)
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(1, Sigmoid)
```

### Output

The sigmoid layer produces:

```text
P(ATTACK)
```

A default classification threshold of `0.5` is used initially.

---

# GRU Architecture

The same general architecture can be implemented using GRU layers:

```text
Input
  ↓
GRU(128)
  ↓
Dropout
  ↓
GRU(64)
  ↓
Dropout
  ↓
Dense(64, ReLU)
  ↓
Dropout
  ↓
Dense(1, Sigmoid)
```

This provides a direct comparison between LSTM and GRU sequence modeling.

---

# Training Strategy

The models use:

### Optimizer

```text
Adam
```

### Loss

```text
Binary Cross-Entropy
```

### Regularization

* Dropout
* Early stopping
* Learning-rate reduction

### Class imbalance

Class weights are calculated from the training sequences:

```python
compute_class_weight(
    class_weight="balanced",
    ...
)
```

This prevents the model from simply favoring the majority class.

---

# Training Callbacks

## Early Stopping

Training stops when validation performance stops improving.

```python
EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)
```

## Reduce Learning Rate

The learning rate is reduced when validation performance plateaus.

```python
ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=2
)
```

---

# Evaluation

Accuracy alone is not sufficient for evaluating an intrusion-detection system.

The project reports:

### Accuracy

Overall proportion of correctly classified samples.

### Precision

Of the traffic predicted as attacks, how much was actually malicious.

### Recall

Of the actual attacks, how many were detected.

For security applications, recall is particularly important because missed attacks represent false negatives.

### F1-score

Harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between benign and malicious traffic across classification thresholds.

### Confusion Matrix

Provides:

```text
                 Predicted
              Benign   Attack
Actual
Benign          TN       FP
Attack          FN       TP
```

---

# Results

> **Results should be updated with the final measured values after model training and evaluation. Do not insert training accuracy as the final test accuracy.**

### Final Test Performance

| Model | Accuracy | Precision | Recall |  F1 | ROC-AUC |
| ----- | -------: | --------: | -----: | --: | ------: |
| LSTM  |      TBD |       TBD |    TBD | TBD |     TBD |
| GRU   |      TBD |       TBD |    TBD | TBD |     TBD |

The final evaluation is performed on the held-out Friday traffic.

---

# Project Structure

Recommended repository structure:

```text
network-intrusion-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lstm_training.ipynb
│   ├── 04_gru_training.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sequence_builder.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── models/
│   ├── final_lstm.keras
│   ├── final_gru.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── app/
│   └── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# Technologies Used

### Programming

* Python

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Visualization

* Matplotlib

### Model Serialization

* Joblib
* Keras model format

### Deployment

* Streamlit / FastAPI *(optional extension)*

### Development Environment

* Google Colab
* Jupyter Notebook
* Git/GitHub

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<username>/<repository>.git

cd <repository>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
joblib
streamlit
```

---

# Running the Project

## 1. Download the dataset

Download CIC-IDS2017 from the official Canadian Institute for Cybersecurity dataset page.

Place the MachineLearningCSV dataset according to the project structure.

---

## 2. Run preprocessing

```bash
python src/preprocessing.py
```

This generates:

```text
processed/
├── training data
├── validation data
├── test data
├── scaler.pkl
└── feature_names.pkl
```

---

## 3. Train LSTM

```bash
python src/train.py --model lstm
```

---

## 4. Train GRU

```bash
python src/train.py --model gru
```

---

## 5. Evaluate

```bash
python src/evaluate.py
```

---

# Inference Pipeline

After training, a new network-flow sample follows:

```text
New Network Flow
       ↓
Feature Validation
       ↓
Feature Ordering
       ↓
Missing Value Handling
       ↓
Saved Scaler
       ↓
Sequence Construction
       ↓
Trained LSTM / GRU
       ↓
Attack Probability
       ↓
Classification
```

Example output:

```text
Prediction: ATTACK
Probability: 0.97
```

or:

```text
Prediction: BENIGN
Probability: 0.08
```

---

# Deployment Architecture

The trained model can be exposed through a REST API or web interface.

```text
              User / Network System
                       │
                       ▼
                  REST API
                       │
                       ▼
              Feature Validation
                       │
                       ▼
                  Preprocessor
                       │
                       ▼
                LSTM / GRU Model
                       │
                       ▼
                Threat Prediction
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          BENIGN                ATTACK
```

A Streamlit interface can additionally provide:

* Flow input
* Attack probability
* Predicted class
* Model confidence
* Prediction history
* Confusion-matrix visualization
* Performance dashboard

---

# Engineering Considerations

## Class imbalance

Network intrusion datasets often contain significantly different numbers of benign and attack samples.

The project addresses this using class-weighted training and evaluation metrics beyond accuracy.

## Generalization

The day-based evaluation strategy is designed to measure performance on traffic from previously unseen dataset files.

## Reproducibility

Random seeds should be fixed when possible:

```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

## Model artifacts

The following artifacts must be versioned or stored alongside the trained model:

```text
model
scaler
feature_names
label mapping
sequence length
classification threshold
```

A model cannot reliably process production traffic without using the same preprocessing configuration used during training.

---

# Limitations

This project uses the CIC-IDS2017 benchmark and therefore has several limitations.

* CIC-IDS2017 is a benchmark dataset rather than live production traffic.
* Performance on CIC-IDS2017 does not guarantee equivalent performance on modern enterprise networks.
* Network-flow features depend on the flow-generation process.
* The initial task focuses on binary intrusion detection.
* Sequence construction depends on the ordering and granularity of available flow records.
* Further validation on external datasets or real network traffic would be required before production deployment.

---

# Future Improvements

Potential extensions include:

### 1. Multiclass Threat Classification

Move from:

```text
BENIGN vs ATTACK
```

to:

```text
BENIGN
DoS
DDoS
PortScan
Botnet
Brute Force
Web Attack
Infiltration
...
```

### 2. Transformer-Based Network Traffic Modeling

Compare recurrent models against Transformer-based sequence architectures.

### 3. Explainable AI

Add feature-level explanations using techniques such as:

* SHAP
* Integrated Gradients
* Feature importance analysis

### 4. Real-Time Detection

Build a streaming inference pipeline:

```text
Network Traffic
      ↓
Flow Extraction
      ↓
Feature Processing
      ↓
Sequence Buffer
      ↓
LSTM/GRU
      ↓
Real-Time Alert
```

### 5. MLOps

Add:

* Experiment tracking
* Model versioning
* Data versioning
* Automated evaluation
* Model monitoring
* Drift detection
* CI/CD

### 6. External Dataset Validation

Evaluate the trained system on additional intrusion-detection datasets to measure cross-dataset generalization.

---

# Skills Demonstrated

This project demonstrates practical experience with:

```text
Python
│
├── Data Engineering
│   ├── Large CSV processing
│   ├── Data cleaning
│   ├── Missing-value handling
│   └── Feature pipelines
│
├── Machine Learning
│   ├── Data splitting
│   ├── Feature scaling
│   ├── Class imbalance
│   └── Evaluation
│
├── Deep Learning
│   ├── RNN
│   ├── LSTM
│   ├── GRU
│   ├── Dropout
│   └── Optimization
│
├── Cybersecurity
│   ├── Network intrusion detection
│   ├── Threat classification
│   └── Attack detection
│
└── ML Engineering
    ├── Reproducible preprocessing
    ├── Model serialization
    ├── Inference pipeline
    └── Deployment architecture
```

---

# What This Project Demonstrates to Recruiters

This project goes beyond simply training an LSTM on a cybersecurity dataset.

It demonstrates the ability to:

* Work with a large real-world cybersecurity dataset
* Build an end-to-end ML pipeline
* Design a leakage-aware evaluation strategy
* Handle imbalanced classification
* Engineer sequential inputs for recurrent neural networks
* Implement and compare LSTM and GRU architectures
* Use validation-driven model selection
* Evaluate models using security-relevant metrics
* Serialize models and preprocessing artifacts
* Design a path toward API/production deployment

---

# Conclusion

The **Sequence-Based Deep Learning System for Network Intrusion Detection and Threat Classification** provides an end-to-end framework for detecting malicious network traffic using recurrent deep learning architectures.

The project combines:

**Cybersecurity + Machine Learning + Deep Learning + Sequence Modeling + ML Engineering**

with an emphasis on **generalization, reproducibility, and deployment readiness** rather than relying solely on raw accuracy.

---

## Author

**Sarfraz Ali Katpar**

BS Computer Science — Sukkur IBA University

**Focus:** Artificial Intelligence | Machine Learning | Deep Learning | NLP | LLMs | AI Engineering

GitHub: `https://github.com/KatparSarfraz1214`

LinkedIn: `https://linkedin.com/in/sarfraz-ali-katpar`

---

## License

This project is intended for educational, research, and portfolio purposes.

The CIC-IDS2017 dataset is subject to its respective terms of use and should be obtained from its official source.
