import pandas as pd

# Define file paths
proteomic_file = "Protein_matrix_averaged_20221214.tsv"
input_file = "input.txt"
output_file = "merged_input_with_proteomics.txt"

# Load proteomic data
print("Loading proteomic data...")
proteomic_df = pd.read_csv(proteomic_file, sep="\t", header=3)
proteomic_df = proteomic_df.set_index(proteomic_df.columns[1])  # Set model_id as index
proteomic_df = proteomic_df.iloc[:, 2:]  # Retain only protein data (exclude metadata)

# Filter proteins with values for all cell lines
proteomic_filtered = proteomic_df.dropna(axis=1, how="any")
num_proteins_retained = proteomic_filtered.shape[1]
print(f"Proteomic data: {num_proteins_retained} proteins with values for all cell lines.")

# Load input data
print("Loading input data...")
input_df = pd.read_csv(input_file, sep="\t")

# Merge input data with proteomic data
print("Merging input data with proteomic data...")
merged_df = input_df.merge(proteomic_filtered, left_on="cell", right_index=True, how="inner")
num_rows_after_merge = merged_df.shape[0]
print(f"Input data: {num_rows_after_merge} rows left after merging.")

# Save the merged dataset
merged_df.to_csv(output_file, sep="\t", index=False)
print(f"Merged data saved to {output_file}.")

