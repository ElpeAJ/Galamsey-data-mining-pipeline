# Galamsey Threat Detection Pipeline: Data Mining & Edge AI Acceleration

An end-to-end artificial intelligence project designed to analyze, model, and accelerate predictive analytics regarding the environmental impact of illegal mining (Galamsey) on regional water bodies. 

This repository showcases the full AI lifecycle, transitioning from exploratory visual data mining (**Orange**) to hardware-level model acceleration (**Intel OpenVINO**).

<img width="250" alt="Orange Data Mining" src="https://github.com/user-attachments/assets/6ebc2213-6441-4408-bd26-b7e5af0366ea" />
<img width="250" alt="OpenVINO" src="https://github.com/user-attachments/assets/4c608236-866d-4df1-8fac-3677a691b5c0" />

---

## 📌 Project Overview
Illegal mining operations severely degrade water quality through heavy pollutant loading. This project aims to identify the exact environmental parameters that determine a water body's susceptibility to Galamsey contamination. 

By mapping **Nemerow’s Pollution Index** and the **Water Quality Index (WQI)**, the pipeline models pollution thresholds to trigger proactive, real-time environmental alerts.

---

## 📊 Phase 1: Exploratory Data Mining & Regression (Orange)
In this phase, raw environmental data matrices are ingested to uncover statistical anomalies and establish a predictive model.

### Workflow Architecture
* **Ingestion:** File node importing multi-parameter water quality data matrices.
* **Feature Selection & Descriptive Analytics:** Filtering parameters using `Feature Statistics` and tracking data spreads through `Distributions`.
* **Predictive Modeling:** Applying a `Linear Regression` node to mathematically model the relationship between Nemerow's Index and overall WQI degradation.

<img width="1507" height="546" alt="Data exploratory - Galamsey" src="https://github.com/user-attachments/assets/985ec69f-3be8-476c-b809-d0ce0270af6e" />

---

## ⚡ Phase 2: High-Speed Model Deployment (Intel OpenVINO)
To scale this project for field deployment (e.g., IoT water sensors or low-power environmental monitoring stations), the model is optimized using the **Intel OpenVINO Toolkit** to minimize processing latency on everyday hardware.

### Key Optimization Features:
* **Model Optimization:** Compiling framework-level tracking structures into OpenVINO Intermediate Representation (`.xml` / `.bin`).
* **Asynchronous Execution:** Utilizing non-blocking inference streams to process continuous incoming sensor arrays in real-time.

---

## How to Run the Project

### Prerequisites
* Orange Data Mining (v3.x or higher)
* Python 3.10+
* Intel OpenVINO Toolkit (`pip install openvino`)

### Running the Data Mining Workspace
1. Launch Orange Data Mining.
2. Open `phase1-orange/data.ows` to interact with the live node pipeline and data tables.

### Executing the OpenVINO Inference Script
Navigate to the OpenVINO directory and run the deployment framework:
```bash
cd phase2-openvino
python inference.py
```
