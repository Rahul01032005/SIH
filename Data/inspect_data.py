import pandas as pd

files = {
    "April": "paimana_month1.csv",
    "May": "paimana_month2.csv",
    "June": "paimana_month3.csv",
    "July": "paimana_month4.csv"
}

print("=== DATA QUALITY AUDIT ACROSS 4 MONTHS ===")
for month, filename in files.items():
    df = pd.read_csv(filename, header=None)
    print(f"\n{month} Report ({filename}):")
    print(f"  - Total Rows: {df.shape[0]}")
    print(f"  - Total Columns: {df.shape[1]}")
    print(f"  - Missing/Null Cells: {df.isnull().sum().sum()}")
    print(f"  - Header Row Preview: {df.iloc[0].values[:4]}")

print("\nAudit complete!")