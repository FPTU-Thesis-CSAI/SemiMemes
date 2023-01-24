from arguments import get_args
import os 
import torch
import clip
from model.clip_info import clip_nms, clip_dim
from model.AutoEncoder import *
from data.semi_supervised_data import DefaultImgTransform, TextTransform
import gradio as gr
import pandas as pd
from torch.nn import Sigmoid


def infer(image, text, shaming, stereotype, objectification, violence, cuda=True):
    sigmoid = Sigmoid()
    
    # labels = df.iloc[3:3+4]
    # text = df.iloc[1]
    
    # print(labels)
    # print(text)
    image = im_transforms(image)
    text = txt_transforms([text])
    
    # print(image.shape)
    # print(text)
    
    text = text['clip_tokens']
    
    image = torch.unsqueeze(image, axis=0)
        
    if cuda:
        images = image.cuda()
        texts = text.cuda()
    
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(texts)
        
        pred = sigmoid(model(image_features, text_features))[0]
    
    confidences = {label_cols[i]: float(pred[i]) for i in range(4)} 
    classes = (pred>0.5).type(torch.int64).detach().cpu().numpy()
        
    return confidences, bool(classes[0] == shaming), bool(classes[1] == stereotype), bool(classes[2] == objectification), bool(classes[3] == violence)

def get_examples(demo_df):
    demo_df['filepath'] = demo_df['file_name'].apply(lambda fname: os.path.join("data/MAMI_processed/images/test/", fname))
    return demo_df[['filepath', 'Text Transcription', *label_cols]].values.tolist()

if __name__ == '__main__':
    args = get_args()

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu
    else:
        cuda = False

    input_resolution = None
    clip_model = None
    cdim = None

    if args.use_clip:
        print("==============use clip model===============")
        # if args.use_open_clip:
        #     clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model,pretrained=args.clip_pretrained)
        #     input_resolution = clip_model.visual.image_size[0]
        # else:
        clip_model, _ = clip.load(clip_nms[args.clip_model],jit=False)
        clip_model = clip_model.float()
        input_resolution = clip_model.visual.input_resolution
        
        cdim = clip_dim[args.clip_model]
        clip_model.eval()
        
        image_size = input_resolution

    if cuda:
        clip_model = clip_model.cuda()
        
    print(clip_model)
    
    
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']


    image_ae = AutoEncoder(encode_image=True)
    text_ae = AutoEncoder(encode_text=True)
    # if cuda:
    #     image_ae.cuda()
    #     text_ae.cuda()       
    
    model = ModelCombineAE(image_encoder=image_ae.encoder, text_encoder=text_ae.encoder)
    
    model.load_state_dict(torch.load(args.saved_model_path))
    
    # print(model)
    
    if cuda:
        model.cuda()     

    clip_model = clip_model.eval()
    model = model.eval()
    
    im_transforms = DefaultImgTransform(img_size=image_size).test_transform
    txt_transforms = TextTransform(args, use_clip=args.use_clip)
    
    demo_df = pd.read_csv('data/MAMI_processed/demo_df.csv')
    

    interface = gr.Interface(fn=infer,
                inputs=[gr.Image(type="pil"), gr.Textbox(visible=False), gr.Number(visible=False), 
                                                                                gr.Number(visible=False), 
                                                                                gr.Number(visible=False), 
                                                                                gr.Number(visible=False)],
                outputs=[gr.Label(num_top_classes=4), gr.Checkbox(label=label_cols[0]), 
                                                        gr.Checkbox(label=label_cols[1]), 
                                                        gr.Checkbox(label=label_cols[2]), 
                                                        gr.Checkbox(label=label_cols[3]),
                                                        ],
                examples=get_examples(demo_df),
                title='FPTU-CSAI')
    
    
    interface.launch(share=True)
    

