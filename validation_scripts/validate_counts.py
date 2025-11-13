#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def load_counts_txt(path):
    """
    Load and aggregate all matrices from a CherryML result.txt file.
    Handles AA-only and AA-3Di matrices.
    """
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip() != ""]

    num_matrices = int(lines[0].split()[0])
    num_states = int(lines[1].split()[0])

    print(f"Loading {num_states}x{num_states} counts from {num_matrices} matrices in {path}...")

    aggregate_df = None
    idx = 3  # the file has 3 header lines (index: 0, 1, 2)
    for m in range(num_matrices):
        col_labels = [x for x in lines[idx].strip().split() if x] # each matrix block starts with a header of the states
        if len(col_labels) != num_states:
            raise ValueError(f"Expected {num_states} column labels, got {len(col_labels)} at matrix {m}.")

        data_lines = lines[idx+1 : idx+1+num_states]  # num_states rows

        row_labels = []
        data = []
        for l in data_lines:
            parts = l.split()
            row_labels.append(parts[0])
            data.append([float(x) for x in parts[1:]])  # all remaining are numeric

        df = pd.DataFrame(data, index=row_labels, columns=col_labels)
        if aggregate_df is None:
            aggregate_df = df
        else:
            aggregate_df += df

        idx += num_states + 1 + 1 # each matrix = 1 line of header + state lines + 1 line of float

    return aggregate_df

    
def marginalize_aa3di_to_aa(df_aa_3di):
    """
    Given aggregated AA-3Di counts (rows/columns like A|A, A|R),
    marginalize to AA-only counts by summing over 3Di states.
    """
    aa_counts = {}
    for row in df_aa_3di.index:
        aa_row = row.split("|")[0]
        for col in df_aa_3di.columns:
            aa_col = col.split("|")[0]
            aa_counts[(aa_row, aa_col)] = aa_counts.get((aa_row, aa_col), 0.0) + df_aa_3di.at[row, col]

    aa_labels = sorted(set([k[0] for k in aa_counts.keys()]))
    data = []
    for r in aa_labels:
        data.append([aa_counts.get((r, c), 0.0) for c in aa_labels])

    df_aa = pd.DataFrame(data, index=aa_labels, columns=aa_labels)
    return df_aa

def compare_counts(df1, df2, top_n=20):
    """
    Compare two AA count matrices: max abs diff, sum abs diff, correlations, top mismatches.
    """
    if set(df1.index) != set(df2.index) or set(df1.columns) != set(df2.columns):
        missing_in_df1 = set(df2.index) - set(df1.index)
        missing_in_df2 = set(df1.index) - set(df2.index)

        msg_lines = ["Error: index/column mismatch detected between matrices."]
        if missing_in_df1:
            msg_lines.append(f"  Missing in marginalized (df1): {sorted(missing_in_df1)}")
        if missing_in_df2:
            msg_lines.append(f"  Missing in AA-only (df2): {sorted(missing_in_df2)}")

        # Join all messages and raise an error
        raise ValueError("\n".join(msg_lines))
    else:
        # Safe path: identical labels
        common_rows = df1.index
        common_cols = df1.columns

    df1_aligned = df1.loc[common_rows, common_cols]
    df2_aligned = df2.loc[common_rows, common_cols]

    flat1 = df1_aligned.values.flatten()
    flat2 = df2_aligned.values.flatten()
    diff = np.abs(flat1 - flat2)
    
    print(f"Max absolute difference: {diff.max()}")
    print(f"Sum of absolute differences: {diff.sum()}")
    print(f"Mean absolute difference per entry: {diff.mean()}")

    spearman_corr, _ = spearmanr(flat1, flat2)
    pearson_corr, _ = pearsonr(flat1, flat2)
    print(f"Spearman correlation (rank) between AA-only and marginalized counts: {spearman_corr}")
    print(f"Pearson correlation (linear) between AA-only and marginalized counts: {pearson_corr}")

    # Top N mismatches
    df_diff = pd.DataFrame(np.abs(df1 - df2), index=df1.index, columns=df1.columns)

    # Compute percent deviation relative to AA-only counts
    df_percent = (df_diff / (df2.replace(0, np.nan))) * 100  # avoid division by zero

    # Flatten both for sorting — prioritize by percent deviation
    df_combined = pd.DataFrame({
        "diff": df_diff.stack(),
        "percent_dev": df_percent.stack()
    }).sort_values(by="percent_dev", ascending=False)

    print(f"\nTop {top_n} mismatched AA pairs (row->col) by percent deviation:")
    for (r, c), row in df_combined.head(top_n).iterrows():
        aa_only = df2.at[r, c]
        marg = df1.at[r, c]
        diff = row["diff"]
        pct = row["percent_dev"]
        print(f"{r}->{c}: AA-only={aa_only}, marginalized={marg}, diff={diff}, deviation={pct:.3f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aa3di_counts_path", help="Path to AA-3Di counts result.txt")
    parser.add_argument("aa_counts_path", help="Path to AA-only counts result.txt")
    args = parser.parse_args()

    counts_aa_3di = load_counts_txt(args.aa3di_counts_path)
    counts_aa = load_counts_txt(args.aa_counts_path)

    counts_aa_by_aa_3di_marg = marginalize_aa3di_to_aa(counts_aa_3di)

    compare_counts(counts_aa_by_aa_3di_marg, counts_aa)
