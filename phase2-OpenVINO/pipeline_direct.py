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

