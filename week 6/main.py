import time
import yaml
import os
import dask.dataframe as dd
from column_name_modification import column_name_modification
from write_yaml_file import yaml_file_Creation

__author__ = 'Madhvik Bhalani'

FILE_NAME = 'data'
FILE_TYPE = 'csv'
INBOUND_DEL= ','
OUTBOUND_DEL= '|'
OUTPUT_FILE = 'pipe_separated_file.gz'

start_time = time.time()

# call write yaml file function
yaml_file_Creation(FILE_NAME,FILE_TYPE,INBOUND_DEL,OUTBOUND_DEL)

#check number of columns and name 
def column_name_validation(df_columns, config_data_columns): 
    if len(df_columns) == len(config_data_columns) and list(config_data_columns)  == list(df_columns):
        return 1
    else:
        return 0

# Read YAML file
with open("file.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    
    file_name = data_loaded['file_name'] + '.' + data_loaded['file_type']
    config_data_columns = data_loaded['columns']

#read file from yaml data
df = dd.read_csv(file_name,sep=INBOUND_DEL)  # dask 
df = column_name_modification(df)  # modify column 
    
# validation of column name 
if column_name_validation(df.columns.tolist(), config_data_columns):  
    print("Columns Validation passed")
else:
    print("Columns Validation failed")
    
   
# Write the DataFrame to a pipe-separated text file in gzipped format
df.to_csv(OUTPUT_FILE, sep=OUTBOUND_DEL, index=False, single_file=True, compression='gzip')

# Read the gzipped pipe-separated text file using Dask
pipe_separated_df = dd.read_csv(OUTPUT_FILE, sep=OUTBOUND_DEL, compression='gzip', blocksize=None) 
# blocksize=None ensures that the entire file is read in one go.

# Display the DataFrame from gzipped pipe-separated text file
print(pipe_separated_df.head())

# File Summary
num_rows = len(df)
num_cols = len(df.columns)
file_size_bytes = os.path.getsize(f'{FILE_NAME}.{FILE_TYPE}')
file_size_mb = file_size_bytes / (1024 * 1024)  # Converting bytes to MB

# Print the results
print("Total number of rows:", num_rows)
print("Total number of columns:", num_cols)
print("File size (in MB):", round(file_size_mb,2))


execution_time = time.time() - start_time
print("--------------------------------")
print(f"Execution Time: {execution_time:.4f} seconds")