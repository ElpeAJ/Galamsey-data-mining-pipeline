# 🌊 Galamsey Threat Detection Pipeline: Data Mining & Edge AI Acceleration
<img width="1560" height="715" alt="Galamsey effect on Water bodies - before & after" src="https://github.com/user-attachments/assets/bc38ec8a-850c-4a35-b6d5-6f77dd4969e7" />
> *Fig. 1 - Before and After picture of ... due to Galamsey on Water bodies*

Illegal mining operations (**Galamsey**) severely degrade regional water bodies through toxic runoff, sediment loading, and destructive chemical dumping. This project establishes an end-to-end Machine Learning lifecycle to model environmental water degradation thresholds and accelerate the resulting predictive processing engine for low-power edge hardware deployment.

This repository demonstrates a complete engineering workflow: transitioning from exploratory data discovery and visual prototyping in **Phase 1 (Orange Data Mining)** to automated training, cross-framework bridging, and high-speed non-blocking deployment in **Phase 2 (Intel OpenVINO)**.
<br>
<img width="350" alt="Orange Data Mining" src="https://github.com/user-attachments/assets/6ebc2213-6441-4408-bd26-b7e5af0366ea" />
<img width="350" alt="OpenVINO" src="https://github.com/user-attachments/assets/4c608236-866d-4df1-8fac-3677a691b5c0" /> <br>
> *Fig.2 & Fig.3 - Logo of Orange Mining Tool and Intel's OpenVINO*

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

<img width="1461" height="856" alt="Screenshot 2026-08-19 at 12 28 31 AM" src="https://github.com/user-attachments/assets/ad1df703-85c9-488a-a2fa-b7091dace951" />
<br>
<!-- <img width="566" height="380" alt="Screenshot 2026-08-19 at 12 11 57 AM" src="https://github.com/user-attachments/assets/4bf3b6a0-20d0-4d04-bcd7-0468fbe07779" /> -->
<!-- <img width="1095" height="824" alt="Screenshot 2026-08-18 at 10 37 43 PM" src="https://github.com/user-attachments/assets/9bba1210-3dba-4de4-9ed2-b83ca231b4b1" /> -->
> *Fig.4 - screenshot of the full, clean node workspace on Orange canvas background.*
<br>

### Configuration & Variable Mapping
* Inside the File grid node configuration settings, `Nemerow_Index` was explicitly defined as the numeric **Feature** (independent variable), and `WQI` was set as the numeric **Target** (dependent variable).
<br>
<img width="780" height="435" alt="Screenshot 2026-08-18 at 10 38 56 PM" src="https://github.com/user-attachments/assets/dc14cb71-f44b-4fa0-9746-0f44fbb55516" />
<br>
> *Fig. 5 - screenshot of the File widget grid window where the feature role was applied to Nemerow_Index and the target role to WQI.*
<br>
* All supplementary raw chemical markers were ignored to prevent collinearity distortion in our simple linear regression.

### The Mathematical Signature

The regression model successfully tracked a near-flawless negative linear relationship with a Pearson Correlation Coefficient of **$r = -0.98$**, calculating a concrete intercept baseline of `99.7930` and a degradation coefficient slope of `-13.3728`. 
<br>
<img width="767" height="485" alt="Screenshot 2026-08-18 at 10 48 34 PM" src="https://github.com/user-attachments/assets/1fc38c9f-8ff8-417d-8453-86ef4230fae9" /> 
<br> 
> *Fig.6 - snapshot of the Linear Regression properties dialog showing the model coefficients and intercepts calculated by Orange.*
<br>
<img width="1504" height="946" alt="Screenshot 2026-08-18 at 10 51 16 PM" src="https://github.com/user-attachments/assets/6496092f-2001-4a40-b106-77adfaf55061" />
<br>
> *Fig.7 - generated scatter plot graph displaying the downward-sloping regression trend line passing smoothly through the 150 data points.*

---

## Phase 2: Edge AI Optimization & Developer Pipeline (Intel OpenVINO)
While visual prototyping tools like Orange provide valuable baseline insights, they cannot run efficiently on resource-constrained, low-power hardware in the field (such as solar-powered river sensor relays floating in water bodies). 

To solve this, we moved our trained model into pure Python code and accelerated it using the **Intel OpenVINO Toolkit**. 

### The Isolated Developer Installation Approach
Instead of implementing heavy, complex, system-wide installation architectures or manual C++ visual installation managers (e.g., standard documentation steps such as `sudo mkdir /opt/intel` or sourcing manual `setupvars.sh` environment paths), we utilized an isolated **Python Developer Pipeline** built directly inside a local virtual environment (`.venv`). By executing:
```bash
pip install openvino
```
Intel's pre-compiled, high-performance hardware-level optimization core libraries were automatically linked directly into our Python runtime environment, completely bypassing the need for desktop installation windows or global file-system modifications.

---

## Script System Architecture & File Walkthrough

The production pipeline is split into modular Python scripts to strictly isolate data engineering tasks from training, compiling, and real-time execution.

<img width="349" height="532" alt="Screenshot 2026-08-18 at 11 17 13 PM" src="https://github.com/user-attachments/assets/a42ea8bd-4135-4044-b5eb-46fac3be6928" />
<img width="349" height="532" alt="Screenshot 2026-08-18 at 11 18 01 PM" src="https://github.com/user-attachments/assets/3c9f93be-4fda-404b-aeaa-faf28543c0de" />

### 1️⃣ Data Telemetry Ingestion Engineering (`generate_data.py`)
Because real-world Galamsey pollution metrics are often heavily incomplete or improperly formatted across public datasets, this script acts as our data ingestion engine. It programmatically generates 150 scientifically realistic environmental water telemetry rows, modeling an inverse relationship with random Gaussian noise to mirror genuine river sensor variances.

```bash
import numpy as np
import pandas as pd

#set random seed for reproducible results
np.random.seed(42)

# Generate 150 random data samples for the Galamsey dataset
# Lowest Nemerow values = cleaner water, highest Nemerow values = more polluted water
nemerow_index =np.random.uniform(0.5, 6.0, 150)


# Calculate WQI with an inverse mathematical relationship + environmental noise
# Clean water approaches 100 WQI; highly contaminated drops towards 10-20 WQI
wqi = 100 - (nemerow_index * 13.5) + np.random.normal(0, 4, 150)
wqi = np.clip(wqi, 10, 100)  # Restrict bounds to logical index scale

# Save out to a clean dataset for your data mining pipeline
df = pd.DataFrame({"Nemerow_Index": nemerow_index, "WQI": wqi})
df.to_csv("galamsey_pollution_data.csv", index=False)
print("Successfully generated clean data file: galamsey_pollution_data.csv!")
```

### The 3 Distinct Model Compilation Strategies Implemented

Depending on one's production requirements, this pipeline supports three distinct execution variants to turn data insights into hardware-accelerated OpenVINO Intermediate Representation (**IR**) deployment files (`.xml` network blueprints and `.bin` parameter arrays):

#### Option A: The Native Code Compiler Pipeline (`pipeline_direct.py`)
This option operates independently of Orange. It ingests your raw CSV dataset, uses Scikit-Learn to train an Ordinary Least Squares linear regression model directly in code, extracts the model's computed parameters, and maps them directly into OpenVINO graph execution operators (`multiply` and `add`).
* **Why it was built:** It provides maximum independence. One can tweak, scale, or expand the data properties programmatically without ever touching a graphical application interface.

```bash
# from openvino import Core
import pandas as pd
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
import openvino as ov

# Initialize OpenVINO Core
core = ov.Core()

# Print available devices (you already verified this outputs ['CPU'])
print("Available devices:", core.available_devices)

# Your data mining pipeline code goes here...
print("Starting data mining pipeline...")
print("------ Step 1: Load the dataset ------")

#1. Load your generated water quality dataset
df = pd.read_csv("phase2-OpenVINO/galamsey_pollution_data.csv")


#2. Define your features (X) and target variable (y)
X = df[["Nemerow_Index"]].values  # Input: Degree of mining runoff pollution
y = df["WQI"].values   # Target: Resulting Water Quality Index

print("------ Step 2: Training Regression Model ------")


#3. Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)
print("Model trained successfully!")
print(f"Mathematical Baseline ==> Model Coefficients: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")


import numpy as np

print("\n--- Step 3: Compiling to OpenVINO Framework ---")

# 1. Reconstruct the regression using native OpenVINO mathematical operators
# Create a symbolic input node representing incoming data stream batches
input_node = ov.opset10.parameter(shape=[-1, 1], dtype=np.float32, name="Nemerow_Index")

# 2. Extract trained parameters from our scikit-learn model object
m_weight = model.coef_.astype(np.float32)       # Slope coefficient
c_intercept = np.array([model.intercept_], dtype=np.float32)  # Intercept constant

# 3. Create OpenVINO constant tensors for the math operators
weight_const = ov.opset10.constant(m_weight, dtype=np.float32)
intercept_const = ov.opset10.constant(c_intercept, dtype=np.float32)

# 4. Construct the prediction graph: WQI = (Nemerow_Index * m) + c
mul_node = ov.opset10.multiply(input_node, weight_const)
output_node = ov.opset10.add(mul_node, intercept_const, name="WQI_Prediction")

# 5. Package into an official OpenVINO Model container structure
ov_model = ov.Model(output_node, [input_node], "Galamsey_WQI_Predictor")

# 6. Save the final optimized OpenVINO Intermediate Representation files
ov.save_model(ov_model, "openvino_model.xml")

print("=" * 50)
print("SUCCESS: Optimized deployment files generated natively!")
print(" -> openvino_model.xml (Describes network topology)")
print(" -> openvino_model.bin (Contains binary weights and parameters)")
print("=" * 50)

```
<br>
<img width="971" height="477" alt="Screenshot 2026-08-18 at 11 34 02 PM" src="https://github.com/user-attachments/assets/47bf3f86-6935-4769-9848-fbfd3ae5a278" />

#### Option B: The Headless Cross-Framework Translation Bridge (`pipeline_orange_bridge.py`)
This script creates a direct translation link between Phase 1 and Phase 2. It opens your saved Orange workspace model binary file (`.pkcls`) using a raw binary stream, extracts the underlying Scikit-Learn model object nested inside, and maps it natively to OpenVINO.
* **Why it was built:** This is a core feature of enterprise data engineering. It allows you to prototype visually in Orange, save your work, and instantly convert that exact model file into high-speed code without any desktop window manager overhead or user interface conflicts.

```bash
#### Refer to phase2-OpenVino/pipeline_orange_bridge.py for full code
# ...

print("\n--- Step 2: Compiling to OpenVINO Framework ---")
# 3. Reconstruct the model structure using OpenVINO graph operators
input_node = ov.opset10.parameter(shape=[-1, 1], dtype=np.float32, name="Nemerow_Index")

# Create OpenVINO constant execution tensors
weight_const = ov.opset10.constant(m_weight, dtype=np.float32)
intercept_const = ov.opset10.constant(c_intercept, dtype=np.float32)

# Reconstruct the prediction logic graph: WQI = (Nemerow_Index * m) + c
mul_node = ov.opset10.multiply(input_node, weight_const)
output_node = ov.opset10.add(mul_node, intercept_const, name="WQI_Prediction")

# ...

```
<br>
<img width="1430" height="483" alt="Screenshot 2026-08-18 at 11 37 10 PM" src="https://github.com/user-attachments/assets/b78cb1fe-6c4b-4dc8-9d18-12224ebefeb8" />


#### Option C: The PyQt/Qt6 Graphics Integration Pipeline (`pipeline_orange_pyqt.py`)
This pipeline uses the full `from Orange.modelling import Model` package library approach, satisfying all underlying framework class bindings by installing desktop dependencies (`PyQt6` and `PySide6`) inside the virtual environment.
* **Why it was built:** This option was implemented to demonstrate how to resolve system environment traps and verify model schemas natively using the parent library's official validation tools.

<img width="1430" height="496" alt="Screenshot 2026-08-18 at 11 41 43 PM" src="https://github.com/user-attachments/assets/8478aee6-b4f8-40c1-b768-41bd029e857e" />
