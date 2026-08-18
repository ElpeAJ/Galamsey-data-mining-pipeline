import os
import pickle
import numpy as np
import openvino as ov

print("--- Step 1: Loading Saved Orange Model (.pkcls) ---")
orange_model_path = "phase1-Orange/galamsey_model_orange.pkcls"

if not os.path.exists(orange_model_path):
    print(f"Error: {orange_model_path} not found! Save it from Orange first.")
    exit()

# 1. Open the file directly using an isolated pickle binary stream
with open(orange_model_path, "rb") as f:
    orange_model = pickle.load(f)

# 2. Extract the underlying Scikit-Learn mathematical model attributes
# This safely pulls the coefficients without triggering Orange's UI modules
sklearn_model = orange_model.skl_model
m_weight = sklearn_model.coef_.astype(np.float32)
c_intercept = np.array([sklearn_model.intercept_], dtype=np.float32)

print("Successfully extracted Scikit-Learn regression parameters without UI overhead.")
print(f"Orange Baseline Model Weights -> Slope (m): {m_weight[0]:.4f}, Intercept (c): {c_intercept[0]:.4f}")

print("\n--- Step 2: Compiling to OpenVINO Framework ---")
# 3. Reconstruct the model structure using OpenVINO graph operators
input_node = ov.opset10.parameter(shape=[-1, 1], dtype=np.float32, name="Nemerow_Index")

# Create OpenVINO constant execution tensors
weight_const = ov.opset10.constant(m_weight, dtype=np.float32)
intercept_const = ov.opset10.constant(c_intercept, dtype=np.float32)

# Reconstruct the prediction logic graph: WQI = (Nemerow_Index * m) + c
mul_node = ov.opset10.multiply(input_node, weight_const)
output_node = ov.opset10.add(mul_node, intercept_const, name="WQI_Prediction")

# 4. Package into an official OpenVINO Model container structure
ov_model = ov.Model(output_node, [input_node], "Galamsey_Orange_Predictor")

# 5. Save the final optimized OpenVINO Intermediate Representation files
# We target the specific phase2 folder path to keep things organized
ov.save_model(ov_model, "phase2-OpenVINO/openvino_model.xml")

print("=" * 60)
print("SUCCESS: Optimized deployment files generated FROM ORANGE MODEL!")
print(" -> phase2-OpenVINO/openvino_model.xml")
print(" -> phase2-OpenVINO/openvino_model.bin")
print("=" * 60)
