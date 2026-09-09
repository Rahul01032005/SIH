import pandas as pd
from src.data.cleaner import clean_dataframe
def test_cleaner_normalizes_and_deduplicates():
    result = clean_dataframe(pd.DataFrame({"Physical Progress %": ["50%", "50%"], "Cost Amount": ["₹100", "₹100"]}))
    assert len(result) == 1 and result.physical_progress.iloc[0] == .5 and result.cost_amount.iloc[0] == 100
