import os
import pickle
import numpy as np
import openvino as ov
from Orange.modelling import Model

print("--- Step 1: Loading Saved Orange Model (.pkcls) ---")
orange_model_path = "phase1-Orange/galamsey_model_orange.pkcls"

if not os.path.exists(orange_model_path):
    print(f"Error: {orange_model_path} not found! Save it from Orange first.")
    exit()

# Open and unpickle the file exported from Orange
with open(orange_model_path, "rb") as f:
    orange_model = pickle.load(f)

# Extract the underlying Scikit-Learn mathematical model from Orange's wrapper
sklearn_model = orange_model.skl_model
print("Successfully extracted Scikit-Learn regression object from Orange model wrapper.")

print("\n--- Step 2: Compiling to OpenVINO Framework ---")
# Initialize OpenVINO core
core = ov.Core()

# Reconstruct the model using standard OpenVINO mathematical operators
input_node = ov.opset10.parameter(shape=[-1, 1], dtype=np.float32, name="Nemerow_Index")

# Extract the trained coefficients directly out of the Orange-trained model object
m_weight = sklearn_model.coef_.astype(np.float32)
c_intercept = np.array([sklearn_model.intercept_], dtype=np.float32)

# Create OpenVINO constant tensors for execution
weight_const = ov.opset10.constant(m_weight, dtype=np.float32)
intercept_const = ov.opset10.constant(c_intercept, dtype=np.float32)

# Reconstruct the prediction logic: WQI = (Nemerow_Index * m) + c
mul_node = ov.opset10.multiply(input_node, weight_const)
output_node = ov.opset10.add(mul_node, intercept_const, name="WQI_Prediction")

# Package into an official OpenVINO Model container structure
ov_model = ov.Model(output_node, [input_node], "Galamsey_Orange_Predictor")

# Save the final optimized OpenVINO Intermediate Representation files
ov.save_model(ov_model, "openvino_model.xml")

print("=" * 60)
print("SUCCESS: Optimized deployment files generated FROM ORANGE MODEL!")
print(" -> openvino_model.xml (Describes network topology)")
print(" -> openvino_model.bin (Contains binary weights and parameters)")
print("=" * 60)
