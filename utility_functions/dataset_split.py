import pandas as pd
from sklearn.model_selection import train_test_split
from ast import literal_eval

df = pd.read_csv("data/final_dataset_prepared_-1_200_7000_100.csv",converters={'tokenized': literal_eval})

# drop the rows with <UNK> as word
df = df[df.word != '<UNK>']


new = df["filename_path"].str.split("/",expand=True)

print("following might change with the change of folder structure")
print("be careful while using this script")
df["filename_path"] = './' + new[12] + '/' + new[13]+ '/' + new[14] + '/' + new[15]

df_train, df_dummy = train_test_split(df,random_state=202,test_size=0.40)
df_val,df_test = train_test_split(df_dummy,random_state=202,test_size=0.50)

print("length of the datset",df_train.shape[0])

print("length of train, validation and test dataset:")
print(df_train.shape[0],df_val.shape[0],df_test.shape[0])

print("total unique words in train, validation and test dataset:")
print(df_train["word"].unique().shape,df_val["word"].unique().shape,df_test["word"].unique().shape)

print("saving train, validation and test dataset.....")

df_train.to_csv("data/train.csv",index=False)
df_val.to_csv("data/val.csv",index=False)
df_test.to_csv("data/test.csv",index=False)