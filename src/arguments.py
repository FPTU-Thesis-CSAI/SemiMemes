import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='memeloss-supervised-rn18-distilbert-fbh-mmhs-memotion', type=str,
                     help="Optional Name of Experiment (used by tensorboard)")
    parser.add_argument('--no-tqdm', action='store_true', help="Disable tqdm and not pollute nohup out")
    parser.add_argument('-data', metavar='DIR', default='/home/viet/SSLMemes/data/memotion_dataset_7k',
                    help='path to dataset')
    parser.add_argument('-dataset-name', default='memotion',
                    help='dataset name', choices=['hatefulmemes', 'harmeme', 'memotion'])
    parser.add_argument('--task', default='b', help='task a or b for memotion')
    # ================================= ARCHITECTURE ================================
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='Image Model Architecture: ')
    parser.add_argument('--txtmodel', default='distilbert-base-uncased', type=str,
                    choices=['distilbert-base-uncased', 'roberta'],
                    help="Text Model used for encoding text")
    parser.add_argument('--out-dim', default=512, type=int, 
                    help='Embedding dimension for modalities (default: 512)')
    parser.add_argument('--projector', default='std', type=str,
                    choices=['std', 'clip', 'pie'], help="Projection used for Unsupervised Training")
    parser.add_argument('--ckpt', default='', type=str,
                    help='Path to load for checkpoint')
    parser.add_argument('--dropout', default=0.2, type=float,
                    help="Dropout probability in classification layer of model")
    parser.add_argument('--fuse_type', default='pie', type=str,
                    choices=['selfattend', 'concat', 'pie', 'mlb'],
                    help="How to combine embeddings in supervised learning")
    parser.add_argument('--nl', default='tanh', type=str,
                    choices=['tanh', 'relu', 'sigmoid'],
                    help="Non Linearity to use between layers of projection heads")
    parser.add_argument('--cl-ckpt', default='', type=str,
                    help="Resume classifier from this checkpoint location")
    parser.add_argument('--num-classes', default=4, type=int,
                    help="Number of Classes in Supervised Setting")
    parser.add_argument('--bn', action='store_true', default=True,
                    help="Use Batch Norm in Classifier")
    # ================================= TRAINING ================================
    parser.add_argument('--dryrun', action='store_true', default=False,
                    help='Use for initial testing purposes only. Runs train for 4 iterations')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')

    parser.add_argument('--gpu-index', default=0, type=int,
                        help='Gpu index.')

    parser.add_argument('--n-samples', default=0, type=int,
                        help="Number of samples used in training")

    parser.add_argument('--vis-embed', action='store_true', default=False,
                        help='Visualise Embeddings in Tensorboard, can not be used with supervised learning.')
    # ================================= EVALUATE ===================================
    parser.add_argument('--evaluate_only', action='store_true', default=False,
                    help="Only evaluate the given model at checkpoints")
    # ================================= LOSSES =====================================
    parser.add_argument('--simclr', action='store_true', help="Use SimCLR for Unsupervised Training on Image Views")
    parser.add_argument('--n-views', default=1, type=int, metavar='N',
                        help='Number of views for contrastive learning training. 1 means no views generated Setting')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--moco_size', default=0, type=int,
                        help="Size of Memory Bank (MoCo, 2020), size=0 is set for SimCLR")
    # Multimodal Contrastive Learning
    parser.add_argument('--mmcontr', action='store_true', help="Use MMContrLoss for Unsupervised Training",default=True)
    parser.add_argument('--measure', default='cosine', type=str,
                        choices=['cosine', 'order'], help="Similarity measure to be used in MMContrLoss")
    parser.add_argument('--margin', default=0, type=float,
                        help="Margin to be used in MMContrLoss")
    parser.add_argument('--max_violation', action='store_true', default=False,
                        help="Consider only the max violation in MMContrLoss")
    
    parser.add_argument('--img_feature_path', type=str,default="data/features/visualgenome/")
    parser.add_argument('--train_csv_path', type=str, default="data/splits/random/memotion_train.csv")
    parser.add_argument('--val_csv_path', type=str, default="data/splits/random/memotion_val.csv")
    parser.add_argument('--model_type', type=str, default="visualbert", help="visualbert or lxmert or vilt")
    parser.add_argument('--use_small_model', type=bool, default=True, help="visualbert or lxmert or vilt")
    # MemeMultimodal Loss
    parser.add_argument('--memeloss', action='store_true', help="Use Meme Multimodal Loss for Unsupervised Training",default=False)
    parser.add_argument('--w-f2i', type=float, default=0.2, help="Fuse2Image Loss Weight")
    parser.add_argument('--w-f2t', type=float, default=0.2, help="Fuse2Text Loss Weight")
    parser.add_argument('--w-f2f', type=float, default=0.6, help="Fuse2Fuse Loss Weight")
    #VLM 
    parser.add_argument('--model_path', type=str, default="uclanlp/visualbert-vqa-coco-pre")
    # parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--amp',type=bool,default=True, \
                help="automatic mixed precision training")
    parser.add_argument('--output_dir', type=str, default="./tmp")
    parser.add_argument('--checkpoint_step', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--semi-supervised', type=bool, default=False)
    parser.add_argument('--use-sweep', type=bool, default=False)
    parser.add_argument('--hyper_yaml_path', type=str, default="config/hyper1.yml") 
    args = parser.parse_args()
    return args