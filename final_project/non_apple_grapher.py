import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FOLDER = "./results/" 

# ==========================================
# 2. DATA LOADING & FILTERING
# ==========================================
all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
df_list = []

print(f"Found {len(all_files)} CSV files.")

for filename in all_files:
    try:
        temp_df = pd.read_csv(filename)
        fname = os.path.basename(filename)
        label = os.path.splitext(fname)[0] # Use filename as label
        
        temp_df['Session'] = label
        df_list.append(temp_df)
        print(f"Loaded: {label}")
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if not df_list:
    print("No CSV files loaded!")
    exit()

df = pd.concat(df_list, ignore_index=True)

# FILTER: REMOVE APPLE DEVICES
# We keep everything that does NOT contain "Apple"
non_apple_df = df[~df['Vendor/Name'].str.contains("Apple", case=False, na=False)]

print(f"\nTotal Devices: {len(df)}")
print(f"Non-Apple Devices: {len(non_apple_df)}")

# ==========================================
# 3. VISUALIZATION
# ==========================================
sns.set_theme(style="whitegrid")

# Helper for styling
def style_plot(title, xlabel, ylabel):
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.tight_layout()

# --- Plot 1: Non-Apple Counts ---
plt.figure(figsize=(12, 6))
counts = non_apple_df.groupby('Session')['MAC Address'].nunique().sort_values(ascending=False)
# Fix for Seaborn warning: use hue=x and legend=False
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="magma", legend=False)
style_plot("Non-Apple Device Count per Session", "Location", "Unique Devices")
plt.show()

# --- Plot 2: Top Non-Apple Vendors ---
plt.figure(figsize=(12, 8))
top_vendors = non_apple_df['Vendor/Name'].value_counts().head(15).index
filtered_vendors = non_apple_df[non_apple_df['Vendor/Name'].isin(top_vendors)]

sns.countplot(data=filtered_vendors, y='Vendor/Name', order=top_vendors, hue='Vendor/Name', palette="viridis", legend=False)
# Manual styling since style_plot assumes x-axis labels are rotated
plt.title("Top 15 Non-Apple Vendors Found", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Count", fontsize=14, fontweight='bold')
plt.ylabel("Vendor", fontsize=14, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# --- Plot 3: Chattiness (Non-Apple) ---
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=non_apple_df, 
    x='Duration Visible (s)', 
    y='Ping Count', 
    hue='Session', 
    style='Session', 
    s=120, 
    alpha=0.8
)
style_plot("Non-Apple Behavior: Duration vs Pings", "Duration Visible (s)", "Ping Count")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()