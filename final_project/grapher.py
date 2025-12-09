import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FOLDER = "./results/" 

LOCATION_MAP = {
    # Add manual mappings here if needed
}

# ==========================================
# 2. DATA LOADING
# ==========================================
all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
df_list = []

print(f"Found {len(all_files)} CSV files.")

for filename in all_files:
    try:
        temp_df = pd.read_csv(filename)
        
        # Extract filename only
        fname = os.path.basename(filename)
        
        # Determine Label
        if fname in LOCATION_MAP:
            label = LOCATION_MAP[fname]
        else:
            # Drop the .csv extension
            label = os.path.splitext(fname)[0]
        
        temp_df['Session'] = label
        temp_df['Source_File'] = fname
        df_list.append(temp_df)
        print(f"Loaded: {fname} -> Label: {label} ({len(temp_df)} devices)")
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if not df_list:
    print("No CSV files loaded! Check your path.")
    exit()
else:
    df = pd.concat(df_list, ignore_index=True)

# --- DATA CLEANING STEP ---
# Map ID: 283 to Hewlett Packard Enterprise
df['Vendor/Name'] = df['Vendor/Name'].replace('ID: 283', 'Hewlett Packard')
print("Replaced 'ID: 283' with 'Hewlett Packard'")

# ==========================================
# 3. METRICS GENERATION
# ==========================================
print("\n" + "="*40)
print("       AUDIT SUMMARY METRICS       ")
print("="*40)

# Metric 1: Total Unique Devices per Session
devices_per_session = df.groupby('Session')['MAC Address'].nunique().sort_values(ascending=False)
print(f"\n--- Device Density (Advertisers / 30s) ---\n{devices_per_session}")

# Metric 2: Apple vs Non-Apple Ratio
df['Is_Apple'] = df['Vendor/Name'].str.contains("Apple", case=False, na=False)
vendor_split = df.groupby('Session')['Is_Apple'].mean() * 100
print(f"\n--- Apple Dominance (% of devices) ---\n{vendor_split.round(1)}")

# Metric 3: Risk Assessment
# Added 'Hewlett' and 'JBL' to risk keywords
risk_keywords = ['Medical', 'ResMed', 'Dexcom', 'Desk', 'TV', 'Sonos', 'Bose', 'Hear', 'Hewlett', 'JBL']
pattern = '|'.join(risk_keywords)
risky_devices = df[df['Vendor/Name'].str.contains(pattern, case=False, na=False)]

print(f"\n--- RISKY / SENSITIVE DEVICES FOUND ({len(risky_devices)}) ---")
if not risky_devices.empty:
    print(risky_devices.groupby(['Session', 'Vendor/Name'])['Last RSSI'].mean().reset_index().to_string(index=False))
else:
    print("None detected.")

# ==========================================
# 4. VISUALIZATION
# ==========================================
sns.set_theme(style="whitegrid")

# Helper function to apply consistent styling
def style_plot(title, xlabel, ylabel):
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.tight_layout()

# --- Plot 1: Device Density ---
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=devices_per_session.index, 
    y=devices_per_session.values, 
    hue=devices_per_session.index, 
    palette="viridis", 
    legend=False
)
style_plot(
    "Traceability: Advertisers Detected per 30s Session", 
    "Location", 
    "Count of Unique Devices"
)
plt.show()

# --- Plot 2: Vendor Composition ---
top_vendors = df['Vendor/Name'].value_counts().nlargest(4).index
df['Vendor_Group'] = df['Vendor/Name'].apply(lambda x: x if x in top_vendors else 'Other')

plt.figure(figsize=(12, 6))
ct = pd.crosstab(df['Session'], df['Vendor_Group'], normalize='index')
ct.plot(kind='bar', stacked=True, colormap='tab10', figsize=(12,6), width=0.8)

plt.title("Identifier Persistence: Vendor Composition", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Location", fontsize=14, fontweight='bold')
plt.ylabel("Proportion of Devices", fontsize=14, fontweight='bold')
plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11)
plt.legend(title='Vendor', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot 3: Signal Strength ---
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df, 
    x='Session', 
    y='Last RSSI', 
    hue='Session', 
    palette="coolwarm", 
    inner="quartile", 
    legend=False
)
plt.axhline(-60, color='red', linestyle='--', alpha=0.5, label='Immediate Proximity (-60)')
style_plot(
    "Proximity Analysis: Signal Strength Distribution", 
    "Location", 
    "RSSI (dBm)"
)
plt.legend(loc='upper left') 
plt.show()

# --- Plot 4: Chattiness (UPDATED BINARY COLORS) ---
plt.figure(figsize=(12, 6))

# Create a clean label column for the legend
df['Device_Category'] = df['Is_Apple'].map({True: 'Apple', False: 'Non-Apple'})

sns.scatterplot(
    data=df, 
    x='Duration Visible (s)', 
    y='Ping Count', 
    hue='Device_Category',  # Use the new label column
    palette={'Apple': 'blue', 'Non-Apple': 'red'}, # FORCE Red and Blue
    style='Session',
    s=120, 
    alpha=0.7
)
style_plot(
    "Surveillance Risk: Device 'Chattiness' vs Persistence", 
    "Duration Visible (seconds)", 
    "Total Packets Broadcasted"
)
plt.legend(title="Device Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()