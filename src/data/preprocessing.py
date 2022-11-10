import numpy as np  
import nltk
nltk.download('punkt')
import re
import pickle 
import shutil
import os
from sentence_transformers import SentenceTransformer
import pandas as pd 

def divide_data(df_train):
    """
    Divide trains dataset to 2 part (labeled data and unlabel data)
    :param df_train: dataframe train
    :return: 2 data frame (labeled data and unlabel data) and create file csv of them
    """ 
    train_2 = df_train.copy()
    train_label = df_train.copy()
    train_un_label = df_train.copy()
    
    train_2["label_merge"] = 'p'
    for i in range(train_2.shape[0]):
        train_2["label_merge"][i] = train_2["humour"][i] + train_2["sarcasm"][i] + train_2["offensive"][i]  + train_2["motivational"][i]

    list_label = train_2["label_merge"].value_counts()
    lst_image = []
    for j in range(len(list_label.index)):
        count = 0
        for i in range(len(train_2["label_merge"])):
            if count <= min(122,list_label[0]) :
              if train_2["label_merge"][i] == list_label.index[j]:
                  k = train_2["image_name"][i]
                  lst_image.append(k)
                  count += 1
            else: 
              break

    for i in train_label["image_name"]:
      if i not in lst_image:
        train_label = train_label.drop(train_label[train_label.image_name == i].index)
    train_label = train_label.reset_index(drop=True)

    for i in train_un_label["image_name"]:
      if i in lst_image:
        train_un_label = train_un_label.drop(train_un_label[train_un_label.image_name == i].index)
    train_un_label = train_un_label.reset_index(drop=True)

    train_label.to_csv("/home/viet/SSLMemes/data/memotion_dataset_7k/labeled_data_train.csv", index = False, encoding = 'utf-8-sig')
    train_un_label.to_csv("/home/viet/SSLMemes/data/memotion_dataset_7k/unlabeled_data_train.csv", index = False, encoding = 'utf-8-sig')

    return train_label , train_un_label

# id - name of file image
def name_of_image (list_id_image_train,number):
    """
    Create file csv: list name image (file .pkl)
    :param df_data: dataframe that need to create file list name image
    :param text: number = 1 -> labeled train, number = 2 -> unlabeled train, number = 3 -> val
    :return: None
    """ 
    id_image_train_int = list_id_image_train
    id_image_train = []
    for i in range(len(id_image_train_int)):
        id_image_train.append(list_id_image_train[i])

    if number == 1:
      output_1 = open('/home/viet/SSLMemes/data/memotion_dataset_7k/list_name_image_train_data_label.pkl', 'wb')
      pickle.dump(id_image_train, output_1)
      output_1.close()
    elif number == 2: 
      output = open('/home/viet/SSLMemes/data/memotion_dataset_7k/list_name_image_train_data_unlabel.pkl', 'wb')
      pickle.dump(id_image_train, output)
      output.close()
    elif number == 3:
      output_val = open('/home/viet/SSLMemes/data/memotion_dataset_7k/list_name_image_val.pkl', 'wb')
      pickle.dump(id_image_train, output_val)
      output_val.close()

    return id_image_train

def label_from_data(df_data,number):
    """
    Create file csv label (file .npy)
    :param df_data: dataframe that need to create file label
    :param text: number = 1 -> labeled train, number = 2 -> unlabeled train, number = 3 -> val
    :return: None 
    """ 

    column = ["image_name","text_corrected","text_ocr"]
    df_data = df_data.copy()
    df_data.drop(column ,axis='columns', inplace=True)

    train_label = []
    for sentence in range(len(df_data)):
        sent_vec = []
        for j in df_data.loc[sentence].tolist():
            if j =="1":
                sent_vec.append(1)
            elif j == "0":
                sent_vec.append(0)
        train_label.append(sent_vec)
    train_label = np.asarray(train_label)

    if number == 1:
        with open('/home/viet/SSLMemes/data/memotion_dataset_7k/label_train.npy', 'wb') as f:
            np.save(f, train_label)
    elif number == 2: 
        with open('/home/viet/SSLMemes/data/memotion_dataset_7k/label_train_unlabel.npy', 'wb') as f:
            np.save(f, train_label)
    elif number == 3:
        with open('/home/viet/SSLMemes/data/memotion_dataset_7k/label_val.npy', 'wb') as f:
            np.save(f, train_label)
    
    return train_label

def clean_dataset(df_train , val, val_label):
    """
    clean_dataset trains and vals dataset, convert to same fomart
    :param train_path: path of file csv train
    :param val_path: path of file csv val
    :param val_label_path: path of file csv label truth of val
    :return: 2 data frame train and val have same format 
    """ 
    column = ["overall_sentiment"]
    df_train.drop(column ,axis='columns', inplace=True)

    df_train['humour'] = df_train['humour'].replace(['funny','very_funny','hilarious','not_funny'],['1','1','1','0'])
    df_train['sarcasm'] = df_train['sarcasm'].replace(['general','very_twisted','twisted_meaning','not_sarcastic'],['1','1','1','0'])
    df_train['offensive'] = df_train['offensive'].replace(['slight','very_offensive','hateful_offensive','not_offensive'],['1','1','1','0'])
    df_train['motivational'] = df_train['motivational'].replace(['motivational','not_motivational'],['1','0'])

    #trích xuất label từ val_label qua val
    val["humour"] = "p"
    val["sarcasm"] = "p"
    val["offensive"] = "p"
    val["motivational"] = "p"

    for i in range(val.shape[0]):
      val["humour"][i] = str(val_label["Labels"][i].split("_")[1][0])
      val["sarcasm"][i] = str(val_label["Labels"][i].split("_")[1][1])
      val["offensive"][i] = str(val_label["Labels"][i].split("_")[1][2])
      val["motivational"][i] = str(val_label["Labels"][i].split("_")[1][3])
    
    column_2 = ["Image_URL"]
    val.drop(column_2 ,axis='columns', inplace=True)

    
    #rename column val dataset follow train dataset
    val.columns = ['image_name', 'text_ocr', 'text_corrected', 'humour', 'sarcasm','offensive',	'motivational']

    df_train.to_csv("/home/viet/SSLMemes/data/memotion_dataset_7k/train.csv", index = False, encoding = 'utf-8-sig')
    val.to_csv("/home/viet/SSLMemes/data/memotion_dataset_7k/val.csv", index = False, encoding = 'utf-8-sig')

    return df_train, val


def list_name_image(train_2):
  list_label = train_2["label_merge"].value_counts()
  lst_image = []
  for j in range(len(list_label.index)):
      count = 0
      for i in range(len(train_2["label_merge"])):
          if count <= min(122,list_label[0]) :
            if train_2["label_merge"][i] == list_label.index[j]:
                k = train_2["image_name"][i]
                lst_image.append(k)
                count += 1
          else: 
            break
  return lst_image

def preprocessing_text(text,number):
    """
    Process text and convert to file csv (.npy)
    :param text: text that need to process in dataframes
    :param text: number = 1 -> labeled train, number = 2 -> unlabeled train, number = 3 -> val
    :return: None 
    """ 
    df_text_train = text
    df_text_train = df_text_train.tolist()

    for i in range(len(df_text_train)):
          df_text_train[i] = str(df_text_train[i])
          df_text_train[i] = re.sub(r'\d+', '', df_text_train[i]) # xóa số
          df_text_train[i] = df_text_train[i].lower()
          df_text_train[i] = re.sub(r'\W',' ',df_text_train [i])
          df_text_train[i] = re.sub(r'\s+',' ',df_text_train [i])
        
    wordfreq = {}
    for sentence in df_text_train:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    import heapq
    most_freq = heapq.nlargest(3000, wordfreq, key=wordfreq.get)

    sentence_vectors_train = []
    for sentence in df_text_train:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors_train.append(sent_vec)
    sentence_vectors_train = np.asarray(sentence_vectors_train)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_embedding = model.encode(df_text_train)

    if number == 1:
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_binary_feature_train_label.npy', 'wb') as f:
        np.save(f, sentence_vectors_train)
    elif number == 2: 
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_binary_feature_train_unlabel.npy', 'wb') as f:
        np.save(f, sentence_vectors_train)
    elif number == 3:
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_binary_feature_val.npy', 'wb') as f:
          np.save(f, sentence_vectors_train)

    if number == 1:
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_sbert_feature_train_label.npy', 'wb') as f:
        np.save(f, sbert_embedding)
    elif number == 2: 
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_sbert_feature_train_unlabel.npy', 'wb') as f:
        np.save(f, sbert_embedding)
    elif number == 3:
      with open('/home/viet/SSLMemes/data/memotion_dataset_7k/text_sbert_feature_val.npy', 'wb') as f:
          np.save(f, sbert_embedding)


def create_foder(path):
  if os.path.exists(path)== True:
    shutil.rmtree(path)
    os.mkdir(path)
  else:
    os.mkdir(path)

def pipeline(train_path,val_path,val_label_path):
  '''
  pipeline for all function pre-processing
  '''
#   train_path = input("Enter train_csv path: ")
#   val_path = input("Enter val_csv path: ")
#   val_label_path = input("Enter val_label_csv_path path: ")

  #input file data
  df_train = pd.read_csv(train_path,index_col = 0)
  val = pd.read_csv(val_path)
  val_label = pd.read_csv(val_label_path)

  #create folder (output)
#   path_folder = "/content/ouput_csv"
#   path_folder_data = "/content/ouput_csv/data"
#   path_folder_data_process = "/content/ouput_csv/data_process"
#   create_foder(path_folder)
#   create_foder(path_folder_data)
#   create_foder(path_folder_data_process)

  # clean dataset
  df_train, val = clean_dataset(df_train , val, val_label)

  # divide dataset
  train_supervise_label , train_un_supervise_label = divide_data(df_train)
  
  # file list image name (.pkl)
  name_of_image(train_supervise_label['image_name'],1)
  name_of_image(train_un_supervise_label['image_name'],2)
  name_of_image(val['image_name'],3)
  
  # file label 
  label_from_data(train_supervise_label,1)
  label_from_data(train_un_supervise_label,2)
  label_from_data(val,3)

  # file feature text
  preprocessing_text(train_supervise_label["text_corrected"],1)
  preprocessing_text(train_un_supervise_label["text_corrected"],2)
  preprocessing_text(val["text_corrected"],3)

if __name__ == '__main__':
    pipeline('/home/viet/SSLMemes/data/memotion_dataset_7k/labels.csv',
    '/home/viet/SSLMemes/data/memotion_test_data/test_data/2000_testdata.csv',
    '/home/viet/SSLMemes/data/memotion_test_data/test_data/Meme_groundTruth .csv')