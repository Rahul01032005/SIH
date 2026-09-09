import pandas as pd
import numpy as np
import re

files = {
    '2026-04': 'paimana_month1.csv',
    '2026-05': 'paimana_month2.csv',
    '2026-06': 'paimana_month3.csv',
    '2026-07': 'paimana_month4.csv'
}

def clean_num(s):
    s = re.sub(r'[^0-9.]', '', str(s))
    if not s:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

dfs = []

for snapshot, filename in files.items():
    raw_df = pd.read_csv(filename, header=None, low_memory=False)
    records = []
    
    for _, row in raw_df.iterrows():
        c1 = str(row[1]) if pd.notna(row[1]) else ''
        match = re.search(r'\b(\d{6})\b', c1)
        if match:
            p_id = match.group(1)
            raw_name = str(row[2]).strip() if len(row) > 2 and pd.notna(row[2]) else np.nan
            
            proj_name = raw_name
            agency = np.nan
            if pd.notna(raw_name):
                proj_name = re.sub(r'^[\'"]+|[\'"]+$', '', proj_name).strip()
                ag_match = re.search(r'^(.*?)\s*\(([^()]+)\)$', proj_name)
                if ag_match:
                    proj_name = ag_match.group(1).strip()
                    agency = ag_match.group(2).strip()

            # Collect all remaining tokens from index 3 onwards
            tokens = []
            for val in row.values[3:]:
                if pd.notna(val):
                    v_str = str(val).strip().strip('\'"')
                    if v_str and v_str not in ['nan', 'None']:
                        tokens.append(v_str)

            # Reconstruct comma-split numbers (e.g. ['3', '577'] -> '3577')
            rebuilt_numbers = []
            curr = ""
            for t in tokens:
                if re.match(r'^\d{1,3}$', t) and curr:
                    curr += t
                else:
                    if curr:
                        rebuilt_numbers.append(clean_num(curr))
                    curr = t
            if curr:
                rebuilt_numbers.append(clean_num(curr))

            rebuilt_numbers = [n for n in rebuilt_numbers if pd.notna(n)]

            # Map the sequence of numbers:
            # If 4 numbers: Orig Cost, Rev Cost, Expenditure, Progress
            # If 3 numbers: Orig Cost, Expenditure, Progress
            # If 2 numbers: Orig Cost, Progress
            orig_cost = np.nan
            rev_cost = np.nan
            cum_exp = np.nan
            prog = np.nan

            if len(rebuilt_numbers) >= 4:
                orig_cost = rebuilt_numbers[0]
                rev_cost = rebuilt_numbers[1]
                cum_exp = rebuilt_numbers[2]
                prog = rebuilt_numbers[3]
            elif len(rebuilt_numbers) == 3:
                orig_cost = rebuilt_numbers[0]
                cum_exp = rebuilt_numbers[1]
                prog = rebuilt_numbers[2]
            elif len(rebuilt_numbers) == 2:
                orig_cost = rebuilt_numbers[0]
                prog = rebuilt_numbers[1]
            elif len(rebuilt_numbers) == 1:
                orig_cost = rebuilt_numbers[0]

            records.append({
                'project_id': p_id,
                'project_name': proj_name,
                'agency': agency,
                'state': np.nan,
                'snapshot_month': snapshot,
                'start_date': np.nan,
                'target_completion_date': np.nan,
                'revised_completion_date': np.nan,
                'original_cost': orig_cost,
                'revised_cost': rev_cost,
                'cumulative_expenditure': cum_exp,
                'physical_progress': prog
            })
            
    df_month = pd.DataFrame(records)
    dfs.append(df_month)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.replace(['-', 'N/A', 'NA', 'not available', 'nan', 'None', ''], np.nan)
combined_df = combined_df.drop_duplicates(subset=['project_id', 'snapshot_month'])

final_columns = [
    'project_id', 'project_name', 'agency', 'state', 'snapshot_month',
    'start_date', 'target_completion_date', 'revised_completion_date',
    'original_cost', 'revised_cost', 'cumulative_expenditure', 'physical_progress'
]
combined_df = combined_df[final_columns]

output_path = 'processed_paimana_projects.csv'
combined_df.to_csv(output_path, index=False)

print("\n=== VALIDATION REPORT ===")
print("1. Number of unique projects in each month:")
for sn in ['2026-04', '2026-05', '2026-06', '2026-07']:
    cnt = combined_df[combined_df['snapshot_month'] == sn]['project_id'].nunique()
    print(f"   Month {sn}: {cnt} projects")

print(f"\n2. Number of rows in final dataset: {len(combined_df)}")

month_counts = combined_df.groupby('project_id')['snapshot_month'].nunique()
common_count = (month_counts == 4).sum()
print(f"3. Number of projects appearing in all four months: {common_count}")

dup_count = combined_df.duplicated(subset=['project_id', 'snapshot_month']).sum()
print(f"4. Duplicate count for project_id + snapshot_month: {dup_count}")

print("\n5. Missing-value count for every column:")
print(combined_df.isnull().sum().to_string())

valid_six_digit = combined_df['project_id'].astype(str).str.match(r'^\d{6}$').all()
print(f"\n6. Confirmation that all project IDs are six-digit PAIMANA IDs: {valid_six_digit}")
print(f"\nDeliverable successfully saved: '{output_path}'")