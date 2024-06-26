import pandas as pd
import os
import glob

base_dir = ''
## This is for English language, edit your paths accordingly
## training data

print('Training data')
data_dir = '....your path/MLS_force_aligned/mls_english/train/'

# get the list of csv files at location base_dir + data_dir having the alignments
csv_paths = glob.glob(base_dir + data_dir + '*/**/*.csv', recursive=True)

print('Number of csv files found: ', len(csv_paths))

# Load the csv files into a dataframe and add a new column for the file name
df = pd.concat([pd.read_csv(f).assign(filename=os.path.basename(f)) for f in csv_paths])


# keep only those rows where the Type is 'words'
df = df[df['Type'] == 'words']

# keep only those rows where df['End'] - df['Begin'] > 0.5
df = df[df['End'] - df['Begin'] > 0.5]

# filter out the words having frequency greater than 50 and less than 5
df_subset = df.groupby("Label").filter(lambda x: len(x) <= 50)
df_subset = df_subset.groupby("Label").filter(lambda x: len(x) >= 5)
df_subset.reset_index(drop=True, inplace=True)

# save the dataframe to a csv file
df_subset.to_csv('train_metadata.csv', index=False)

print('Number of rows in the dataframe: ', len(df_subset))
print('Number of unique words: ', len(df_subset['Label'].unique()))

## dev data
print('Dev data')
data_dir = '....your path/MLS_force_aligned/mls_english/dev/'

# get the list of csv files at location base_dir + data_dir having the alignments
csv_paths = glob.glob(base_dir + data_dir + '*/**/*.csv', recursive=True)

print('Number of csv files found: ', len(csv_paths))

# Load the csv files into a dataframe and add a new column for the file name
df = pd.concat([pd.read_csv(f).assign(filename=os.path.basename(f)) for f in csv_paths])

# keep only those rows where the Type is 'words'
df = df[df['Type'] == 'words']

# keep only those rows where df['End'] - df['Begin'] > 0.5
df = df[df['End'] - df['Begin'] > 0.5]

# reset the index and save the dataframe to a csv file
df.reset_index(drop=True, inplace=True)
df.to_csv('dev_metadata.csv', index=False)

print('Number of rows in the dataframe: ', len(df))
print('Number of unique words: ', len(df['Label'].unique()))

## test data
print('Test data')
data_dir = '....your path/MLS_force_aligned/mls_english/test/'

# get the list of csv files at location base_dir + data_dir having the alignments
csv_paths = glob.glob(base_dir + data_dir + '*/**/*.csv', recursive=True)

print('Number of csv files found: ', len(csv_paths))

# Load the csv files into a dataframe and add a new column for the file name
df = pd.concat([pd.read_csv(f).assign(filename=os.path.basename(f)) for f in csv_paths])

# keep only those rows where the Type is 'words'
df = df[df['Type'] == 'words']

# keep only those rows where df['End'] - df['Begin'] > 0.5
df = df[df['End'] - df['Begin'] > 0.5]

# reset the index and save the dataframe to a csv file
df.reset_index(drop=True, inplace=True)
df.to_csv('test_metadata.csv', index=False)

print('Number of rows in the dataframe: ', len(df))
print('Number of unique words: ', len(df['Label'].unique()))

print('Done!')


