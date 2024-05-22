def column_name_modification(df):
    df.columns.tolist()

    new_column_list = []

    for column in df.columns.tolist():
        new_column_list.append(''.join(final_column_name for final_column_name in column if final_column_name.isalnum()).lower())

    df.columns = new_column_list
    
    return df