**Getting Started**

***first clone and access the repository***
```bash
git clone https://github.com/xiaoyuanzi22333/COMP4901_Final.git

cd COMP4901_Final
```

***prepare the dataset***

you can directly downlaod it from https://www.kaggle.com/datasets/sohaibanwaar1203/image-depth-estimation and put them in ./dataset


***prepare the package***
```bash
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

pip install -q roboflow supervision

wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

```

you can also try to download a SAM model you want on their website: https://github.com/facebookresearch/segment-anything.git

***start training***
```bash
normal CNN model:
    python baseline_train.py

SAM enhanced CNN model:
    pythn SAM_training.py
```