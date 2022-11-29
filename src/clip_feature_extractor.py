from data.semi_supervised_data import create_dataloader_clip_extractor
import os
import random
import torch
import numpy as np
import clip
from arguments import get_args
from model.clip_info import clip_nms, clip_dim
from tqdm import tqdm

def extract(model, dataloader, cuda=False):
    image_features_list = []
    text_features_list = []
    
    for images, texts in tqdm(dataloader, total=len(dataloader)):
        texts = texts['clip_tokens']
        
        if cuda:
            images = images.cuda()
            texts = texts.cuda()
        
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features_list.append(image_features)
            text_features_list.append(text_features)        
    
    image_features_arr = torch.cat(image_features_list).cpu().numpy()
    text_features_arr = torch.cat(text_features_list).cpu().numpy()    
    return image_features_arr, text_features_arr    

def save(args, image_features_arr, text_features_arr):
    parent_dir = os.path.basename(args.input_file_clip_extractor).replace('.csv', '')
    root_dir = os.path.join(args.output_dir_clip_extractor, parent_dir)
    os.makedirs(root_dir, exist_ok=True)
    
    np.savetxt(f'{root_dir}/image_feature.txt', image_features_arr)
    np.savetxt(f'{root_dir}/text_feature.txt', text_features_arr) 
    
    return 0

if __name__ == '__main__':
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu
    else:
        cuda = False

    input_resolution = None
    clip_model = None
    cdim = None

    clip_model, _ = clip.load(clip_nms[args.clip_model],jit=False)
    clip_model = clip_model.float()
    input_resolution = clip_model.visual.input_resolution
        
    cdim = clip_dim[args.clip_model]
    clip_model.eval()
        
    image_size = input_resolution

    if cuda:
        clip_model = clip_model.cuda()
        
    loader = create_dataloader_clip_extractor(args,
                                            input_img_dir_clip_extractor=args.input_img_dir_clip_extractor,
                                            input_file_clip_extractor=args.input_file_clip_extractor,
                                            batch_size=args.batchsize, 
                                            image_size=image_size)

    image_features_arr, text_features_arr = extract(clip_model, loader, cuda)
    
    save(args, image_features_arr, text_features_arr)