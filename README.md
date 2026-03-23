# Zero-Shot Coreset Selection

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

([Brent A. Griffin](https://github.com/griffbr)\*, [Jacob Marks](https://github.com/jacobmarks), [Jason J. Corso](https://github.com/jasoncorso)) @ [Voxel51](https://voxel51.com)

\* Corresponding author

**Z**ero-Shot **Core**set Selection ([ZCore](https://arxiv.org/pdf/2411.15349 "ZCore Paper")) is a method of coreset selection for unlabeled data. Deep learning methods rely on massive data, resulting in substantial costs for storage, annotation, and model training. Coreset selection aims to select a subset of the data to train models with lower cost while ideally performing on par with the full data training. Although the majority of real-world data are unlabeled, previous state-of-the-art coreset methods cannot select data that are unlabeled. As a solution, ZCore addresses the problem of coreset selection without labels _or_ training on candidate data. Instead, ZCore uses existing foundation models to generate a zero-shot embedding space for unlabeled data, then quantifies the relative importance of each example based on overall coverage and redundancy within the embedding distribution. On ImageNet, the ZCore coreset achieves a higher accuracy than previous label-based coresets at a 90% prune rate, while removing annotation requirements for 1.15 million images.

__Zero-Shot Coreset Selection Overview__
![alt text](./figure/main_figure.jpg?raw=true "ZCore Overview")

## Using ZCore
We provide example ZCore commands for coreset selection and subsequent model training for the EuroSAT10 dataset from our paper. See instructions in **Repeat Trials** to repeat experiment trials and **Dataset Setup** for full ImageNet, CIFAR, or EuroSAT setup.

Step 1. **Dataset**. [Download](https://www.dropbox.com/scl/fo/1mhwsunssr6g2v1wio0vq/AEI2cx3aZ2vWvFmSLDfUHtQ?rlkey=kbxo4uae43tnzvk6k7x5hk28u&st=8tkh3oyl&dl=0 "EuroSAT split download") and unzip ``eurosat10.zip`` in ``./data``.

Step 2. **Zero-Shot Coreset Selection**
```bash
python zeroshot_coreset_selection.py --dataset eurosat10 --data_dir ./data --results_dir ./results --embedding clip resnet18 --num_workers 10
```
[FiftyOne](https://github.com/voxel51/fiftyone) dependency to generate embeddings (``pip install fiftyone``).

Step 3. **Train Coreset Model**
```bash
python train_coreset_model.py --prune_rate 0.7 --dataset eurosat10 --data_dir ./data --score_file ./results/eurosat10/zcore-eurosat10-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-0/score.npy
```

## Repeat Trials
We provide examples scripts to repeat ZCore experiments over multiple trials in `./repeat-trial-scripts`.

Repeat **ZCore Selections** for EuroSAT10
```bash
chmod +x ./repeat-trial-scripts/eurosat10-score-x5.sh
./repeat-trial-scripts/eurosat10-score-x5.sh
```

Repeat **Coreset Model Training** for EuroSAT10
```bash
chmod +x ./repeat-trial-scripts/eurosat10-train-x5.sh
./repeat-trial-scripts/eurosat10-train-x5.sh
```

We provide example repeat trial results in `./results/example/eurosat10`. To tabulate these repeat trials run:
```bash
python process_repeat_trials.py --base_score_dir ./results/example/eurosat10/zcore-eurosat10-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex
```
to generate the following table:
```
Setting p30-s51 p50-s51 p70-s51 p80-s51 p90-s51 

Trial Results
0       93.80   91.93   86.10   80.98   63.63   
1       93.39   91.26   85.74   78.88   65.58   
2       93.63   91.21   87.91   79.84   66.70   
3       93.90   92.38   86.91   79.86   65.16   
4       94.06   92.26   86.47   80.20   67.75   

Aggregate Results
Mean    93.76   91.81   86.63   79.95   65.76   
StdDev  0.230   0.491   0.750   0.677   1.398   
Overall Mean: 83.58 
```

## Datasets

**ImageNet** can be downloaded [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data "ImageNet download") and subsequently reformatted using:
```bash
cd ./ILSVRC/Data/CLS-LOC/val/                                                               
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

**CIFAR10** and **CIFAR100** can be downloaded [here](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR download").

**EuroSAT80**, **EuroSAT40**, **EuroSAT20**, and **EuroSAT10** can be downloaded [here](https://www.dropbox.com/scl/fo/1mhwsunssr6g2v1wio0vq/AEI2cx3aZ2vWvFmSLDfUHtQ?rlkey=kbxo4uae43tnzvk6k7x5hk28u&st=8tkh3oyl&dl=0 "EuroSAT split download").

## Citation

If you find this code useful, please consider citing our [paper](https://arxiv.org/pdf/2411.15349):

```bibtex
@InProceedings{Griffin_2026_WACV,
  author={Griffin, Brent A. and Marks, Jacob and Corso, Jason J.},
  title={Zero-Shot Coreset Selection via Iterative Subspace Sampling},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

You may also want to check out our open-source toolkit, [FiftyOne](https://voxel51.com/fiftyone), which provides a powerful interface for exploring, analyzing, and visualizing datasets for computer vision and machine learning.
