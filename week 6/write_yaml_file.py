import dask.dataframe as dd
import yaml
from column_name_modification import column_name_modification 

def yaml_file_Creation(FILE_NAME,FILE_TYPE,INBOUND_DEL,OUTBOUND_DEL):
    
    df = dd.read_csv(f'{FILE_NAME}.{FILE_TYPE}')  # dask 
    df = column_name_modification(df)  # modify column 
    
    # Create the dictionary with all the required information
    data = {
        'file_name': FILE_NAME,
        'file_type': FILE_TYPE,
        'inbound_delimiter': INBOUND_DEL,
        'outbound_delimiter': OUTBOUND_DEL,
        'skip_leading_rows': 1,
        'columns': df.columns.tolist()  # Add the column names to the dictionary
    }

    # Write the dictionary to a YAML file
    with open('file.yaml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)