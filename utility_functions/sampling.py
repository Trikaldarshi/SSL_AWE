import pandas as pd
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import operator
from ast import literal_eval
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def find_token(df1,token):
    return token in df1

def update_dist(df,token,num_del):
    k = df["tokenized"].apply(find_token,token=token)
    df = df.drop(df.index[k][:num_del])
    return df

def update_dist1(df,token,num_del):
    k = df["tokenized"].apply(find_token,token=token)
    index_values = df.index[k][:num_del]
    deleted_rows = df.loc[index_values]
    df = df.drop(index_values)
    return df,deleted_rows

def update_dist2(df,list_tokens):
    k = df["tokenized"].apply(find_token,token=list_tokens[0])
    word = df[k]["word"].value_counts().idxmax()
    index_value = df.index[df['word'] == word][0]
    deleted_row = df.loc[index_value]
    df = df.drop(index_value)
    return df,deleted_row

def update_dist_opt(df,token,num_deletion):
    k = df["tokenized"].apply(find_token,token=token)
    words = df[k]["word"].value_counts()[-num_deletion:]
    index_values = []
    for m1,m2 in words.items():
        index_values.append(df.index[df['word'] == m1][0])
    deleted_rows = df.loc[index_values]
    df = df.drop(index_values)
    return df,deleted_rows

def get_spk_id_and_ch_id(df):
    return "-" + "-".join(df.split(sep = "_")[0].split(sep="-")[1:])
def match_string(target_string,df2):
    matched_string = df2.loc[df2['path'].str.contains(target_string, case=False)]
    if len(matched_string)==0:
        return np.nan
    return matched_string.values[0][0]

# df = pd.read_csv("dataset_prepared200.csv",converters={'tokenized': literal_eval})

def sampling(df,num_sampling, num_deletion, plot=True):
    print("total unique words to be sampled", num_sampling)
    print("num of deletions to be performed at one go:", num_deletion)

    token_list = []
    for i in df['tokenized'].values:
        token_list += i

    dict_tokens = Counter(token_list)
    print("total tokens before sampling:",len(dict_tokens))

    if plot==True:
        labels, values = zip(*Counter(token_list).items())

        indexes = np.arange(len(labels))
        width = 1
        plt.rcParams["figure.figsize"] = (100,20)
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig('../data/before_norm.pdf')
        plt.close()

    dict_tokens = dict(dict_tokens)

    mean_df = int(sum(dict_tokens.values())/len(dict_tokens))

    print("mean frequency before sampling", mean_df)

    # np.std(np.array([*dict_tokens.values()]))

    # tokens_sorted = dict(sorted(dict_tokens.items(), key=operator.itemgetter(1),reverse=False)).keys()
    print("removing the tokens with frequency less than 1000:")
    while min(dict_tokens.values())<1000:
        for token in  dict(sorted(dict_tokens.items(), key=operator.itemgetter(1),reverse=False)).keys():
            # print(token,dict_tokens[token])
            if dict_tokens[token] < 1000:
                num_del = dict_tokens[token]
                # print(token,num_del)
                df,_ = update_dist1(df,token,num_del)
            else:
                break
            token_list = []
            for i in df['tokenized'].values:
                token_list += i

            dict_tokens = Counter(token_list)
            dict_tokens = dict(dict_tokens)


    token_list = []
    for i in df['tokenized'].values:
        token_list += i

    dict_tokens = Counter(token_list)
    print("total tokens after removing tokens with freq <1000:",len(dict_tokens))

    if plot==True:
        labels, values = zip(*Counter(token_list).items())

        indexes = np.arange(len(labels))
        width = 1
        plt.rcParams["figure.figsize"] = (100,20)
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig('../data/before_norm_clipped.pdf')
        plt.close()

    # dict_tokens.most_common()

    # max_unique = df["word"].unique().shape

    factor=1.0/sum(dict_tokens.values())
    for k in dict_tokens:
        dict_tokens[k] = 1 - dict_tokens[k]*factor

    df_uniform = pd.DataFrame(columns=df.columns)

    # len(df_uniform["word"])

    # df.shape
    random.seed(202)
    count = 0
    # num_deletion = 500
    while len(df_uniform["word"].unique()) < num_sampling:
        sample_token = random.choices(population=list(dict_tokens.keys()),weights=list(dict_tokens.values()),k=1)[0]

        df,deleted_rows = update_dist_opt(df,sample_token,num_deletion=num_deletion)
        df_uniform = df_uniform.append(deleted_rows)
        print("sampled dataset length",df_uniform.shape[0])
        
        token_list = []
        for i in df['tokenized'].values:
            token_list += i
        
        dict_tokens = Counter(token_list)
        factor=1.0/sum(dict_tokens.values())
        for k in dict_tokens:
            dict_tokens[k] = 1 - dict_tokens[k]*factor
        count = len(df_uniform["word"].unique())
        # if count%100==0:
        print("total unique sampled words ...", count)


    token_list_uniform = []
    for i in df_uniform['tokenized'].values:
        token_list_uniform += i

    dict_tokens_uniform = Counter(token_list_uniform)
    print("total tokens after sampling:",len(dict_tokens_uniform))

    if plot==True:
        labels, values = zip(*Counter(token_list_uniform).items())

        indexes = np.arange(len(labels))
        width = 1
        plt.rcParams["figure.figsize"] = (100,20)
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig('../data/after_norm.pdf')
        plt.close()
    return df_uniform

