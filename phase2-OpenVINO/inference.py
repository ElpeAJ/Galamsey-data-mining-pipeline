import os
import time
import numpy as np
import pandas as pd
import openvino as ov

print("--- Step 1: Initializing Asynchronous OpenVINO Runtime Engine ---")
core = ov.Core()

# Load and compile the architecture
model_xml = "openvino_model.xml"
model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")

# 1. Initialize an AsyncInferQueue for non-blocking execution streams
# We allocate 2 parallel request slots to compute streams concurrently
infer_queue = ov.AsyncInferQueue(compiled_model, jobs=2)

print("--- Step 2: Preparing Environmental Alert Ledger System ---")
ledger_file = "pollution_alerts.csv"

# Create a clean ledger file with headers if it doesn't already exist
if not os.path.exists(ledger_file):
    df_ledger = pd.DataFrame(columns=["Timestamp", "Sensor_ID", "Nemerow_Index", "Predicted_WQI", "Status"])
    df_ledger.to_csv(ledger_file, index=False)

print("\n--- Step 3: Stream Processing Data Ingestion Pipeline ---")
# 2. Ingest the actual data matrix asset you generated earlier
data_source = "phase2-OpenVINO/galamsey_pollution_data.csv"
if not os.path.exists(data_source):
    print(f"Error: {data_source} missing! Run generate_data.py first.")
    exit()

df_stream = pd.read_csv(data_source)
print(f"Loaded continuous data telemetry matrix containing {len(df_stream)} rows.")

print("=" * 80)
print(f"{'Stream ID':<12}{'Nemerow Index':<18}{'Predicted WQI':<18}{'Pipeline Deployment Status'}")
print("=" * 80)

# 3. Define the Asynchronous Callback Function
# This function triggers automatically on a background thread the split-second a prediction completes
def completion_callback(request, userdata):
    # Extract structural identifiers passed down through userdata
    stream_id, nemerow_val, start_time = userdata
    latency = (time.perf_counter() - start_time) * 1000
    
    # Extract the resulting calculated prediction vector tensor
    output_layer = request.get_output_tensor(0)
    predicted_wqi = float(output_layer.data[0][0])
    
    # Threshold classification rules for field monitoring telemetry
    if predicted_wqi >= 70:
        status = "🟢 SAFE / EXCELLENT"
    elif 50 <= predicted_wqi < 70:
        status = "🟡 WARNING: Moderate Pollution"
    else:
        status = "🔴 CRITICAL: Heavy Galamsey Contamination!"
        
        # 4. Append severe environmental violations to the tracking ledger file immediately
        alert_entry = pd.DataFrame([{
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Sensor_ID": f"SNS-{stream_id:03d}",
            "Nemerow_Index": round(nemerow_val, 4),
            "Predicted_WQI": round(predicted_wqi, 4),
            "Status": "CRITICAL G_RUNOFF"
        }])
        alert_entry.to_csv(ledger_file, mode='a', header=False, index=False)
        
    print(f"Telemetry #{stream_id:<4} | {nemerow_val:<14.2f} | {predicted_wqi:<14.2f} | {status} ({latency:.2f}ms)")

# Link our background thread callback mechanism to the inference queue handlers
infer_queue.set_callback(completion_callback)

# 5. Push data matrices through the non-blocking execution thread engine
for idx, row in df_stream.head(20).iterrows():  # We process the first 20 records as a live demonstration stream
    nemerow_input = float(row["Nemerow_Index"])
    
    # Format structural input data into standard OpenVINO 2D input layout
    input_tensor = np.array([[nemerow_input]], dtype=np.float32)
    
    # Pack tracking identifiers into user payload packages
    tracking_payload = (idx + 1, nemerow_input, time.perf_counter())
    
    # Push into the queue layer non-blockingly; execution transfers to background worker threads instantly
    infer_queue.start_async({0: input_tensor}, userdata=tracking_payload)
    
    # Throttle slightly to simulate real-world sensor communication delays
    time.sleep(0.1)

# Wait completely until all asynchronous background threads finish processing remaining queue payloads
infer_queue.wait_all()

print("=" * 80)
print(f"SUCCESS: Continuous stream pipeline completed.")
print(f"Environmental warning entries have been securely archived inside: '{ledger_file}'")
