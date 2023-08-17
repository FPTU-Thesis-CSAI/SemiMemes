import pandas as pd 
import argparse
import random 
import os 

parser = argparse.ArgumentParser(description='train_split')
parser.add_argument('--train_csv_path', type=str, default="data/MAMI_processed/train.csv")
parser.add_argument('--label_ratio', type=float, default=0.03)
parser.add_argument('--save_dir', type=str, default="data/MAMI_processed")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

df = pd.read_csv(args.train_csv_path)
# id_samples = df['Id'].tolist()
# labeled_id_sample = random.sample(id_samples,int(args.label_ratio*len(id_samples)))
# unlabeled_id_sample = list(set(id_samples).difference(set(labeled_id_sample)))

# labeled_sample = {"labeled_id":labeled_id_sample}
# unlabeled_sample = {"unlabeled_id":unlabeled_id_sample}

# df1 = pd.DataFrame(labeled_sample) 
# df2 = pd.DataFrame(unlabeled_sample)

df_labeled = df.sample(frac=args.label_ratio, random_state=args.seed)
df_unlabeled = df.drop(df_labeled.index)

df_labeled.to_csv(os.path.join(args.save_dir,f"train_labeled_ratio-{args.label_ratio}.csv"), index=False)
df_unlabeled.to_csv(os.path.join(args.save_dir,f"train_unlabeled_ratio-{args.label_ratio}.csv"), index=False)