# Slim Scissors: Segmenting Thin Object from Synthetic Background
This is the official implementation of our work `Slim Scissors: Segmenting Thin Object from Synthetic Background`.

> **Slim Scissors: Segmenting Thin Object from Synthetic Background**,            
> Kunyang Han, Jun Hao Liew, Jiashi Feng, Huawei Tian, Yao Zhao*, Yunchao Wei  
> In: European Conference on Computer Vision (ECCV), 2022
> [[pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890375.pdf)] [[supplementary](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890375-supp.pdf)]

## Installation

0. Clone the repo:
    ```
    git clone https://github.com/KunyangHan/SlimScissors
    cd SlimScissors
    ```

1. Install dependencies.
    ```
    pip install -r requirements.txt
    ```

2. Download the dataset by running the script inside ```data/```:
    ```.bash
    cd data/
    chmod +x download_dataset.sh
    ./download_dataset.sh
    cd ..
    ```
    Data folder expects the following structure:

    ```Shell
    data
    ├── COIFT
    │   ├── images
    │   ├── list
    │   └── masks
    ├── HRSOD
    │   ├── images
    │   ├── list
    │   └── masks
    ├── ThinObject5K
    │   ├── images
    │   ├── list
    │   └── masks
    └── thin_regions
        ├── coift
        │   ├── eval_mask
        │   └── gt_thin
        ├── hrsod
        │   ├── eval_mask
        │   └── gt_thin
        └── thinobject5k_test
            ├── eval_mask
            └── gt_thin
    ```

1. Download pretrain weights
    ```
    wget https://download.pytorch.org/models/resnet18-5c106cde.pth
    ```

## Training
We provide the scripts for training our models on ThinObject-5K dataset. You can start training with the following commands:
```.bash
python train.py --data_root path_to_data_folder
```

## Testing
Download checkpoint by [Google Drive](https://drive.google.com/file/d/1U8LY5aSxPhzOPReg1jWxaXXG1gC1AotI/view?usp=sharing).

To evaluate, simply run following commands:
```.bash
python evaluation.py --ckpt path_to_pth_file --data_root path_to_data_folder --dataset target_dataset
```

For example
```.bash
python evaluation.py --ckpt ckpt/checkpoint_epoch_29.pth --data_root ../data --dataset HRSOD
```


## License
This project is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/legalcode).


## Citation
If you use this code, please consider citing our paper:

    @InProceedings{han2022slim,
        title     = {Slim Scissors: Segmenting Thin Object from Synthetic Background},
        author    = {Han, Kunyang and Liew, Jun Hao and Feng, Jiashi and Tian, Huawei and Zhao, Yao and Wei, Yunchao},
        booktitle = {eccv},
        year      = {2022},
    }
