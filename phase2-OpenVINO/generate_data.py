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
