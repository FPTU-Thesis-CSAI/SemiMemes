# SSLMemes
Multi-modal Semi-supervised Learning for Sentiment Analysis of Internet Memes
[[Paper]] (https://arxiv.org/abs/2304.00020)

# Abstract

The prevalence of memes on social media has created the need to sentiment analyze their underlying meanings for censoring harmful content. Meme censoring systems by machine learning raise the need for a semi-supervised learning solution to take advantage of the large number of unlabeled memes available on the internet and make the annotation process less challenging. Moreover, the approach needs to utilize multimodal data as memes' meanings usually come from both images and texts. This research proposes a multimodal semi-supervised learning approach that outperforms other multimodal semi-supervised learning and supervised learning state-of-the-art models on two datasets, the Multimedia Automatic Misogyny Identification and Hateful Memes dataset. Building on the insights gained from Contrastive Language-Image Pre-training, which is an effective multimodal learning technique, this research introduces SemiMemes, a novel training method that combines auto-encoder and classification task to make use of the resourceful unlabeled data.

# Algorithm Overview 

![SemiMemes](overallAchitecture.png)

# How to train?

Install pytorch version compatible with your cuda version (https://pytorch.org/get-started/previous-versions/)

Install dependencies 

```bash
$ pip install -r requirements.txt 
```

To train the main model. We need to run the pretraiing stage and then fintune it.

```bash
$ cd src
$ python pretrain.py
$ python train_2_stages.py
```

# Results from the paper

Weighted-average F1-Measure on Validation and Test Set of MAMI dataset.

Model                  | Val (5%) | Test (5%) | Val (10%) | Test (10%) | Val (30%) | Test (30%)
----------------------- | :------: |  :------: | :------: | :------: | :------: | :------: |
SemiMemes               | **0.693** | **0.6782** | **0.7258** | **0.7113** | **0.7520** | **0.7413**
CMML-CLIP               | 0.6778 |0.6438 | 0.717 | 0.6878 | 0.7313 | 0.7242
TIB-VA                  | 0.68 |0.6392 | 0.6992 | 0.6886 | 0.7095 | 0.7104


AUROC on Dev Seen and Test Seen Set of Hateful Memes.

Model                  | Trainable params | Dev (5%) | Test (5%) | Dev (10%) | Test (10%) | Dev (30%) | Test (30%)
----------------------- | :------: | :------: |  :------: | :------: | :------: | :------: | :------: |
SemiMemes               | 3.1M | **0.6897** | **0.7011** | **0.7061** | **0.7281** | **0.7399** | **0.7765**
Hate-CLIPper (cross)    | 1.5B | 0.6652 | 0.6973 | 0.6827 | 0.7196 | 0.7030 | 0.7731

# How to cite?

```
@misc{tung2023semimemes,
      title={SemiMemes: A Semi-supervised Learning Approach for Multimodal Memes Analysis}, 
      author={Pham Thai Hoang Tung and Nguyen Tan Viet and Ngo Tien Anh and Phan Duy Hung},
      year={2023},
      eprint={2304.00020},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```