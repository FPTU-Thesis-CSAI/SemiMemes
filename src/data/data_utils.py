import pandas as pd
import torch 

def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_.reset_index()

def calculate_label_ratio(label_csv, unlabel_csv):
    label_df = pd.read_csv(label_csv)
    unlabel_df = pd.read_csv(unlabel_csv)

    return len(label_df)/(len(label_df)+len(unlabel_df))

def get_texts(label_csv, unlabel_csv, text_col):
    label_text = pd.read_csv(label_csv)[text_col].to_list()
    unlabel_text = pd.read_csv(unlabel_csv)[text_col].to_list()

    # print(len(label_text+unlabel_text))

    return label_text + unlabel_text

# print(calculate_label_ratio('data/MAMI_processed/train_labeled_ratio-0.3.csv', 'data/MAMI_processed/train_unlabeled_ratio-0.3.csv'))
# get_all_train_text('data/MAMI_processed/train_labeled_ratio-0.3.csv', 'data/MAMI_processed/train_unlabeled_ratio-0.3.csv', text_col='Text Transcription')
