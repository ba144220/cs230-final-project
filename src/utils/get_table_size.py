import pandas as pd
from io import StringIO
from typing import List, Tuple, Dict, Any
import numpy as np
import csv

from transformers import AutoTokenizer


def convert_csv_string_to_table(csv_string: str) -> List[List[str]]:
        """
        Convert a csv string to a table, including the header
        """
        df = pd.read_csv(StringIO(csv_string), delimiter=",", on_bad_lines="skip")
        columns = df.columns.astype(str).tolist()
        columns = [col.replace("Unnamed: 0", "") for col in columns]
        rows = df.values.astype(str).tolist()
        return [columns] + rows
     

def calculate_tokenized_table_size(
    table_string: str,
    tokenizer: AutoTokenizer
) -> Tuple[int, int]:
    """
    Calculate the tokenized table size. Return the number of rows and the width in the number of tokens.
    """
    try:
        table = convert_csv_string_to_table(table_string)
    except:
        print(table_string)
        return 1000000, 1000000
    
    # Find the max length for each column
    token_lengths = np.zeros((len(table), len(table[0])), dtype=int)
    tokenized_table = []
    for i in range(len(table)):
        tokenized_row = []
        for j in range(len(table[0])):
            tokenized_cell = tokenizer.encode(table[i][j], add_special_tokens=False)
            tokenized_row.append(tokenized_cell)
            token_lengths[i, j] = len(tokenized_cell)
        tokenized_table.append(tokenized_row)
    
    # Find the max length for each column
    max_lengths = np.max(token_lengths, axis=0)
    tokenized_table_width = np.sum(max_lengths).item() + max_lengths.shape[0]
    return len(table), tokenized_table_width