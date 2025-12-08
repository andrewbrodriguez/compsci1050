import pandas as pd
import glob
import os

# CONFIGURATION
DATA_FOLDER = "./results/"  # Folder where your CSVs are

def main():
    # 1. Load all files
    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not files:
        print("No CSV files found.")
        return

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not df_list:
        return

    # 2. Combine and Filter
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Filter OUT "Apple" to see the interesting stuff
    others = full_df[~full_df['Vendor/Name'].str.contains("Apple", case=False, na=False)]
    
    # 3. Count and Display
    counts = others['Vendor/Name'].value_counts()
    
    print("\n" + "="*40)
    print("      NON-APPLE DEVICE IDENTIFIER")
    print("="*40)
    print(f"Total 'Other' Devices: {len(others)}")
    print(f"Unique Vendors: {len(counts)}\n")
    
    print(f"{'COUNT':<8} | {'VENDOR / NAME'}")
    print("-" * 40)
    for name, count in counts.items():
        print(f"{count:<8} | {name}")
    print("-" * 40)

if __name__ == "__main__":
    main()