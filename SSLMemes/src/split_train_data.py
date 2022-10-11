import pandas as pd 
import argparse
import random 
import os 

parser = argparse.ArgumentParser(description='train_split')
parser.add_argument('--train_csv_path', type=str, default="data/splits/random/memotion_train.csv")
parser.add_argument('--label_ratio', type=float, default=0.3)
parser.add_argument('--save_dir', type=str, default="data/splits/random")
args = parser.parse_args()

df = pd.read_csv(args.train_csv_path)
id_samples = df['Id'].tolist()
labeled_id_sample = random.sample(id_samples,int(args.label_ratio*len(id_samples)))
unlabeled_id_sample = list(set(id_samples).difference(set(labeled_id_sample)))

labeled_sample = {"labeled_id":labeled_id_sample}
unlabeled_sample = {"unlabeled_id":unlabeled_id_sample}

df1 = pd.DataFrame(labeled_sample) 
df2 = pd.DataFrame(unlabeled_sample) 
df1.to_csv(os.path.join(args.save_dir,"labeled_sample.csv"))
df2.to_csv(os.path.join(args.save_dir,"unlabeled_sample.csv")) 