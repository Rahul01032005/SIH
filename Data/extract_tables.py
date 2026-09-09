import os
import pdfplumber
import pandas as pd

# 1. Define paths to your raw PDFs using the correct 'Raw' folder name
pdf_files = {
    "paimana_month1.csv": "Raw/FlashReport_April2026.pdf",
    "paimana_month2.csv": "Raw/FlashReport_May_2026.pdf",
    "paimana_month3.csv": "Raw/FlashReport_June_2026.pdf",
    "paimana_month4.csv": "Raw/FlashReport_July_2026.pdf"
}

# 2. Extract tables page by page
for csv_name, pdf_path in pdf_files.items():
    if not os.path.exists(pdf_path):
        print(f"Skipping {pdf_path} — file not found.")
        continue
    
    print(f"Extracting tables from {pdf_path}...")
    all_rows = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    all_rows.append(row)
                    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_name, index=False, header=False)
    print(f"Successfully saved: {csv_name} (Total rows: {len(df)})\n")

print("All extractions completed successfully!")