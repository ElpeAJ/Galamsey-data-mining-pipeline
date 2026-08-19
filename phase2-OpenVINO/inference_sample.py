import time
import numpy as np
import openvino as ov

print("--- Step 1: Initializing OpenVINO Runtime Engine ---")
core = ov.Core()

# Load the compiled OpenVINO IR files
model_xml = "openvino_model.xml"
model = core.read_model(model=model_xml)

# Compile the model specifically for your MacBook Pro's active device ('CPU')
compiled_model = core.compile_model(model=model, device_name="CPU")

# Extract the input and output layer keys for data mapping
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print(input_layer)
print(output_layer)
print("Model loaded and optimized successfully on device: CPU\n")

print("--- Step 2: Simulating Live Field Sensor Array Data Stream ---")
# Simulated live stream of Nemerow Pollution Index readings coming from the river site
# Values range from clean water (~0.5) to extreme Galamsey runoff (~5.5)
mock_sensor_stream = [0.8, 1.5, 3.2, 5.1, 2.4]

print(f"Incoming Sensor Batch Queue: {mock_sensor_stream}\n")
print("--- Step 3: Executing Edge Pipeline Inference Loop ---")
print("|" + "=" * 80 + "|")
print(f"{'Reading ID':<12}{'Nemerow Index':<18}{'Predicted WQI':<18}{'Environmental Status'}")
print("|" + "=" * 80 + "|")

for idx, sample in enumerate(mock_sensor_stream, start=1):
    # Format the single input number into a structural 2D float32 array [-1, 1]
    input_tensor = np.array([[sample]], dtype=np.float32)
    
    # Run the inference
    start_time = time.perf_counter()
    inference_result = compiled_model([input_tensor])
    latency = (time.perf_counter() - start_time) * 1000 # Calculate latency in milliseconds
    
    # Extract the predicted Water Quality Index value
    predicted_wqi = inference_result[output_layer][0][0]
    
    # Define Threshold Logic for automated field alerts based on standard WQI rules
    if predicted_wqi >= 70:
        status = "🟢 SAFE / EXCELLENT"
    elif 50 <= predicted_wqi < 70:
        status = "🟡 WARNING: Moderate Pollution"
    else:
        status = "🔴 CRITICAL: Heavy Galamsey Contamination!"
        
    print(f"Sensor #{idx:<5} | {sample:<14.2f} | {predicted_wqi:<14.2f} | {status} (Latency: {latency:.3f}ms)")
    time.sleep(0.5) # Simulate time gap between sensor transmissions

print("|" + "=" * 80 + "|")
print("Data stream inference loop completed successfully.")
