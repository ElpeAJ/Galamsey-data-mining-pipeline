# 🌊 Galamsey Threat Detection Pipeline: Data Mining & Edge AI Acceleration

Illegal mining operations (known locally in Ghana as **Galamsey**) severely pollute our water bodies with toxic runoff, mud, and chemicals. This project is a complete artificial intelligence pipeline built to automatically detect how polluted a river or stream is based on field parameters, and flag environmental alerts in real time.

Think of this repository as a two-stage journey: we start with **visual data exploration (Phase 1)** to figure out the math behind water pollution, and finish by creating a **high-speed automated tracking engine (Phase 2)** designed to run on tiny field computers sitting right next to the river.

---

## 📌 The Core Problem We Are Solving

When looking at a polluted river, environmentalists use two major math scales to track damage:
1. **Nemerow’s Pollution Index:** A score that measures the severity of heavy pollutants present in the water.
2. **Water Quality Index (WQI):** A universal grade for water health. Clean, drinkable water scores near **100**, while heavily contaminated water drops toward **10**.

**Our AI Goal:** Create a system that looks at the Nemerow Index and instantly calculates the resulting Water Quality Index—triggering an automated warning system if conditions become critical.

---

## 📊 Phase 1: Prototyping & Finding the Math Pattern (Orange)

Before writing any complex code, we needed to see if our data made logical sense. We used a visual tool called **Orange Data Mining** to look at our data matrix and train a baseline **Linear Regression** model. 

* **Analogy:** Linear Regression is simply drawing a "line of best fit" through data points. It tells us exactly how much the Water Quality Index drops every time the mining pollution index increases.

> 📷 **[INSERT YOUR ORANGE WORKSPACE SCREENSHOT HERE]**
> *Tip: Put an image showing your Data Table -> Select Columns -> Linear Regression node canvas layout here to visually anchor your workflow.*

---

## ⚡ Phase 2: Building the High-Speed Edge Engine (Intel OpenVINO)

While visual tools like Orange are great for prototyping, they cannot run efficiently on low-power hardware in the field (like a solar-powered water sensor box floating in a river). 

To solve this, we moved our trained math model into pure Python code and accelerated it using the **Intel OpenVINO Toolkit**.

### 🛠️ Step-by-Step Code Walkthrough (For Future Reference)

If you need to recall how this system works down the road, here is the exact operational lifecycle of the code:

### 1️⃣ Step 1: Generating Realistic Sensor Data (`generate_data.py`)
Because field data is incredibly hard to find, we wrote a script that simulates a live river telemetry stream. It generates 150 unique water samples showing a realistic inverse relationship: as mining pollution spikes, water safety metrics fall.

### 2️⃣ Step 2: Training & Hardware Optimization (`pipeline.py`)
This script reads our generated data file and uses Python to train our linear regression math model. It extracts the raw mathematical formulas (the slope weights and intercept constants) and compiles them natively into **OpenVINO Intermediate Representation (IR)** files:
* `openvino_model.xml`: The blueprint describing the model structure.
* `openvino_model.bin`: The compressed mathematical weights.

By saving the model in this format, it can execute math equations on everyday hardware with zero wasted computing power.

### 3️⃣ Step 3: Non-Blocking Asynchronous Execution (`inference.py`)
This is the live production script. It loads the optimized model onto your computer's CPU and sets up a high-performance **Asynchronous Queue**. 
* **Why Async matters:** Instead of processing one sensor reading and freezing while it calculates, the script runs calculations on a background thread. It can process a continuous pipeline of incoming sensor data arrays simultaneously with **zero lag**.
* **Automated Safety Ledger:** Every time a calculated Water Quality score falls below 50, the script automatically generates a warning and permanently logs a timestamped incident entry into an archive file called `pollution_alerts.csv`.

> 📷 **[INSERT YOUR VS CODE TERMINAL RUN SCREENSHOT HERE]**
> *Tip: Use a screenshot showing your terminal processing Telemetry data streams with 0.20ms latencies and logging critical galamsey warnings.*

---

## 🚀 How to Run the Entire Pipeline on Your Machine

Follow these rapid terminal steps to watch the system run from scratch:

### ⚙️ Prerequisites
Ensure you are working inside your Python virtual environment:
```bash
source .venv/bin/activate
```

### 🏃‍♂️ Execution Sequence

1. **Generate the Water Telemetry Matrix:**
   ```bash
   python phase2-OpenVINO/generate_data.py
   ```
2. **Train the Model and Compile the OpenVINO IR Files:**
   ```bash
   python phase2-OpenVINO/pipeline.py
   ```
3. **Launch the Live Asynchronous Stream and View Alert Ledgers:**
   ```bash
   python phase2-OpenVINO/inference.py
   ```

Open your project sidebar after running the code; you can open `pollution_alerts.csv` to check the automated history logs created by your background engine!
