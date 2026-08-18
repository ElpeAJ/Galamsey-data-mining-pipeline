# 🌊 Galamsey Threat Detection Pipeline: Data Mining & Edge AI Acceleration
<img width="1560" height="715" alt="Galamsey effect on Water bodies - before & after" src="https://github.com/user-attachments/assets/bc38ec8a-850c-4a35-b6d5-6f77dd4969e7" />

Illegal mining operations (**Galamsey**) severely degrade regional water bodies through toxic runoff, sediment loading, and destructive chemical dumping. This project establishes an end-to-end Machine Learning lifecycle to model environmental water degradation thresholds and accelerate the resulting predictive processing engine for low-power edge hardware deployment.

This repository demonstrates a complete engineering workflow: transitioning from exploratory data discovery and visual prototyping in **Phase 1 (Orange Data Mining)** to automated training, cross-framework bridging, and high-speed non-blocking deployment in **Phase 2 (Intel OpenVINO)**.
<br>
<img width="250" alt="Orange Data Mining" src="https://github.com/user-attachments/assets/6ebc2213-6441-4408-bd26-b7e5af0366ea" />
<img width="250" alt="OpenVINO" src="https://github.com/user-attachments/assets/4c608236-866d-4df1-8fac-3677a691b5c0" />

---

## Architectural Core: The Math Behind the Pipeline

To accurately quantify and automate environmental devastation assessments, our pipeline models the core relationship between two vital environmental metrics:
1. **Nemerow’s Pollution Index (NPI):** An index tracking the aggregate and maximum concentrations of heavy chemical pollutants present in a specific water sample.
2. **Water Quality Index (WQI):** A universal scale grading overall water health. Perfectly clean, pristine water scores near **100**, while heavily contaminated water drops sharply toward **10**.

### The Mathematical Target Formula
Our objective is to train a high-performance linear model ($WQI = m \cdot NPI + c$) that accepts raw Nemerow pollution data feeds and instantly predicts the resulting Water Quality Index score, triggering automated field alerts when safety parameters drop below acceptable limits.

---

## Phase 1: Prototyping & Visual Modeling (Orange)

Before deploying raw code pipelines, it is crucial to discover data trends and validate the mathematical viability of our targets. We used **Orange Data Mining** to establish our architectural baseline, completely bypassing traditional spreadsheet layout, text-row pollution, or data transposition errors by feeding our clean data matrix asset directly into an OLS (Ordinary Least Squares) Linear Regression model.
<img width="1095" height="824" alt="Screenshot 2026-08-18 at 10 37 43 PM" src="https://github.com/user-attachments/assets/9bba1210-3dba-4de4-9ed2-b83ca231b4b1" />

### Configuration & Variable Mapping
* Inside the File grid node configuration settings, `Nemerow_Index` was explicitly defined as the numeric **Feature** (independent variable), and `WQI` was set as the numeric **Target** (dependent variable).
  <img width="780" height="435" alt="Screenshot 2026-08-18 at 10 38 56 PM" src="https://github.com/user-attachments/assets/dc14cb71-f44b-4fa0-9746-0f44fbb55516" /> <br><br>
* All supplementary raw chemical markers were ignored to prevent collinearity distortion in our simple linear regression.

### The Mathematical Signature

The regression model successfully tracked a near-flawless negative linear relationship with a Pearson Correlation Coefficient of **$r = -0.98$**, calculating a concrete intercept baseline of `99.7930` and a degradation coefficient slope of `-13.3728`. <br>

<img width="767" height="485" alt="Screenshot 2026-08-18 at 10 48 34 PM" src="https://github.com/user-attachments/assets/1fc38c9f-8ff8-417d-8453-86ef4230fae9" /> <br> <br>
<img width="1504" height="946" alt="Screenshot 2026-08-18 at 10 51 16 PM" src="https://github.com/user-attachments/assets/6496092f-2001-4a40-b106-77adfaf55061" />

---
