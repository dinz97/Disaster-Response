# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
        messages_filepath - File path to disaster_messages.csv
        categories_filepath - File path to disaster_categories.csv
    OUTPUT:
        df - Merged dataset of messages data and their categories
    """
    
    # Import disaster messages and their categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
                             
    # Merge messages and categories dataset
    df = messages.merge(categories, on='id')
                       
    return df



def clean_data(df):
    """
    INPUT:
        df - Merged dataset of messages data and their categories
    OUTPUT:
        cleansed_df - cleansed dataset with binary values for response categories columns
    """
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
                       
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories.
    # One way is to apply a lambda function that takes everything 
    # Up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
                       
    # Rename the columns of 'categories'
    categories.columns = category_colnames
       
    # Convert category values to just number 0 or 1
    for column in categories:
                       # Set each value to be the last character of the string
                       categories[column] = categories[column].astype('str').str.split('-').str[1]
                       
                       # Convert column from string to numeric
                       categories[column] = pd.to_numeric(categories[column])
    # Drop the original categories column from df
    df.drop(columns='categories', axis=1, inplace=True)
                       
    # Concatenate the original dataframe with the new 'categories' dataframe
    cleansed_df = pd.concat([df, categories], axis=1)
    
    # Remove rows with any category indicators more than 1
    cleansed_df = cleansed_df[~(cleansed_df[category_colnames] > 1).any(1)]
                       
    # Remove duplicates
    cleansed_df.drop_duplicates(inplace=True)
                       
    # Remove rows with values > 1
    cleansed_df = cleansed_df[~(cleansed_df[category_colnames] > 1).any(1)]
 
    return cleansed_df                   


def save_data(df, database_filepath):
    """
    INPUT:
        df - Merged dataset of messages data and their categories
        database_filepath - Path to database
    """
    # Create SQL engine
    engine = create_engine('sqlite:///' + database_filepath)
                       
    # SQL database
    database_filename = database_filepath.split('/')[1]
    database_filename = database_filename.replace('.db', '')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
