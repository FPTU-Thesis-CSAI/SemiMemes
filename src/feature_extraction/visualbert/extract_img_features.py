from distutils.log import debug
import os
import glob
import numpy as np
from tqdm.auto import tqdm
import argparse

from utils import *
from utils import extract_visual_features

def extract_features(img_folder, cfg, model, batch_size=16):

    # load images
    print (f"load images...")
    all_img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    names, images = [], []
    for i, path in tqdm(enumerate(all_img_paths)):
        #if i ==100: break
        name = path.split("/")[-1]
        img = cv2.cvtColor(plt.imread(path), cv2.COLOR_RGB2BGR)
        names.append(name)
        images.append(img)

    # encode images
    print (f"encode images...")
    visual_embeds_all = []
    for i in tqdm(np.arange(0, len(images), batch_size)):
        visual_embeds = extract_visual_features(cfg, visual_model, images[i:i+batch_size])
        visual_embeds_all += visual_embeds
    visual_embeds_all = torch.stack(visual_embeds_all, dim=0)
    print (f"visual embedding shape:{visual_embeds_all.shape}, # names: {len(names)}")

    return names, visual_embeds_all

def extract_features(all_img_paths, cfg, model, batch_size=1):

    names, images = [], []
    for i, path in tqdm(enumerate(all_img_paths)):
        #if i ==100: break
        name = path.split("/")[-1]
        img = cv2.cvtColor(plt.imread(path), cv2.COLOR_RGB2BGR)
        names.append(name)
        images.append(img)

    # encode images
    print (f"encode images...")
    visual_embeds_all = []
    for i in tqdm(np.arange(0, len(images), batch_size)):
        print(len(images))
        visual_embeds = extract_visual_features(cfg, visual_model, images[i:i+batch_size], debug=True)
        visual_embeds_all += visual_embeds
    visual_embeds_all = torch.stack(visual_embeds_all, dim=0)
    print (f"visual embedding shape:{visual_embeds_all.shape}, # names: {len(names)}")

    return names, visual_embeds_all

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='compute Detectron2 image embeddings')
    parser.add_argument('--img_folder', type=str, required=False, default='data/Memotion2.0/images/train_images')
    parser.add_argument('--output_folder_path', type=str, required=False, default='data/splits/random/memotion_train.csv')

    args = parser.parse_args()

    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = load_config_and_model_weights(cfg_path)
    visual_model = get_model(cfg).cuda()
    img_folder = args.img_folder
    # names, features = extract_features(img_folder, cfg, visual_model, batch_size=1)
    names, features = extract_features(all_img_paths=glob.glob(os.path.join(img_folder, "*.jpg"))[:40], cfg=cfg, model=visual_model, batch_size=1)
    assert len(names) == features.shape[0]
    
    # write out
    if not os.path.isdir(args.output_folder_path):
        os.mkdir(args.output_folder_path)

    output_path_txt = os.path.join(args.output_folder_path, "names.txt")
    output_path_feature = os.path.join(args.output_folder_path, "features.pt")
    # write names to txt
    with open(output_path_txt, "w") as f:
        for name in names:
            f.write(name + "\n")

    # write features to pth
    torch.save(features, output_path_feature)

