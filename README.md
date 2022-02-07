# Video Classification 

## Supported Models & Datasets

Action Recognition

* [X3D](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html) (CVPR 2020) (Coming Soon)
* [UniFormer](https://arxiv.org/abs/2201.04676) (ICLR 2022)

Action Detection

* [SlowFast](https://arxiv.org/abs/1812.03982) (ICCV 2019) (Coming Soon)

Datasets

* [Kinetics-400/600/700](https://deepmind.com/research/open-source/kinetics/) (CVPR 2017)
* [AVA](https://research.google.com/ava/index.html) (CVPR 2018) (Coming Soon)

## Pre-trained Models & Benchmarks

Refer to [MODELS.md](./docs/MODELS.md) for supported pre-trained models and benchmarks of SOTA models.

## Installation

* python >= 3.6
* torch >= 1.9.0
* torchvision >= 0.10.0

Then, clone and install this repo with:

```bash
$ git clone https://github.com/sithu31296/video-classification
$ cd video-classification
$ pip install -e .
```

## Inference

Use the following script to test the pre-trained model:

```bash
$ python tools/infer.py \
    --source VIDEO_FILE_NAME
    --model MODEL_NAME
    --model_path PRETRAINED_MODEL_PATH
    --num_classes DATASET_NUM_CLASSES
```

You will get the top-5 score similar to this:

```
Class            Score (%)
-------------  -----------
archery              91.49
throwing axe          0.11
slacklining           0.06
feeding fish          0.06
rock climbing         0.05
```

## Training & Evaluation

### Dataset Preparation

Follow the steps provided in [DATASETS.md](./docs/DATASETS.md).

### Configuration

Coming Soon

### Training

Coming Soon

### Evaluation

Coming Soon

## References

* [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast/)

## Citations

```
@misc{li2022uniformer,
      title={UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning}, 
      author={Kunchang Li and Yali Wang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
      year={2022},
      eprint={2201.04676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```