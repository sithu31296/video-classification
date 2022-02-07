# Pre-trained Models and Benchmarks

## Kinetics

Model | Pretrain | CropsxClips | FramesxSampling | K400 Top-1 | K600 Top-1 | Params | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | --- | ---
UniFormer-S | IN1K | 1x4 | 8x8 | 78.4 | - | - | 70 | [k400][uniformers8k400]
UniFormer-S | IN1K | 1x4 | 16x4 | 80.8 | 82.8 | - | 167 | [k400][uniformers16k400]/[k600][uniformers16k600]
UniFormer-B | IN1K | 1x4 | 8x8 | 79.8 | - | - | 161 | [k400][uniformerb8k400]
UniFormer-B | IN1K | 1x4 | 16x4 | 82.0 | 84.0 | - | 387 | [k400][uniformerb16k400]/[k600][uniformerb16k600]
UniFormer-B | IN1K | 1x4 | 32x4 | 82.9 | 84.5 | - | 1036 | [k400][uniformerb32k400]/[k600][uniformerb32k600]
||
MViT-B | IN1K | 1x5 | 16x4 | 78.4 | - | - | - | -
MViT-B | IN1K | 1x5 | 32x3 | 80.4 | 83.9 | - | - | - 
||
X3D-S | - | 1x10 | 13x6 | 73.1 | - | 4 | 2 | -
X3D-M | - | 1x10 | 16x5 | 75.1 | - | 4 | 5 | -
X3D-L | - | 1x10 | 16x5 | 76.9 | - | 6 | 18 | -


## Something-Something

Model | Pretrain | CropsxClips | FramesxSampling | SSV1 Top-1 | SSV2 Top-1 | Params | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | --- | ---
UniFormer-S | K400 | 3x1 | 16x4 | 57.2 | 67.7 | - | 125 | [ssv1][uniformers16ssv1]/[ssv2][uniformers16ssv2]
UniFormer-S | K400 | 3x1 | 32x4 | 58.8 | 69.0 | - | 329 | [ssv1][uniformers32ssv1]/[ssv2][uniformers32ssv2]
UniFormer-B | K400 | 3x1 | 16x4 | 59.1 | 70.4 | - | 290 | [ssv1][uniformerb16ssv1]/[ssv2][uniformerb16ssv2]
UniFormer-B | K400 | 3x1 | 32x4 | 60.9 | 71.1 | - | 777 | [ssv1][uniformerb32ssv1]/[ssv2][uniformerb32ssv2]

## AVA

Model | Pretrain | FramesxSampling | mAP | Weights
--- | --- | --- | --- | ---
Slow-R50 | K400 | 4x16 | 19.5 | -


[uniformers8k400]: https://drive.google.com/file/d/1-kitVTkWErHXI_x-sLItvK9-sOmb53j6/view?usp=sharing
[uniformers16k400]: https://drive.google.com/file/d/1-jGZIIuTM5IYIpqkNTojhGhfMc7Gu9Kt/view?usp=sharing
[uniformerb8k400]: https://drive.google.com/file/d/1hBxGU-QE8hGRBjSNCymJ7Au06LhluvTv/view?usp=sharing
[uniformerb16k400]: https://drive.google.com/file/d/15ipzFaZdTt-aHv7jcr_h_vgYD2X7D6AD/view?usp=sharing
[uniformerb32k400]: https://drive.google.com/file/d/1gHxx8cr1H0CNvngj1kB9ArgcPfJtqjCV/view?usp=sharing
[uniformers16k600]: https://drive.google.com/file/d/1-dqzjm5RZVspWHQLRD4S1vo4_6jyVltb/view?usp=sharing
[uniformerb16k600]: https://drive.google.com/file/d/1nBQCBCsCflhZ4OGUoGB3pAkc_GDau7vd/view?usp=sharing
[uniformerb32k600]: https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing
[uniformers16ssv1]: https://drive.google.com/file/d/1-bdFsmXt2Ztf43lrVgmqS8_VYJ8tub6J/view?usp=sharing
[uniformers32ssv1]: https://drive.google.com/file/d/1-Y5ADssqM2C8o_1jix48fQM788mVGWQ3/view?usp=sharing
[uniformerb16ssv1]: https://drive.google.com/file/d/1uMq4W5lf17vOHUmccNjYLeTMqaWHSh6v/view?usp=sharing
[uniformerb32ssv1]: https://drive.google.com/file/d/18A3FuUUbdQ8ehMQ0D78qTTtm6YmIVWaO/view?usp=sharing
[uniformers16ssv2]: https://drive.google.com/file/d/1-Wq22024HB7tT8kPX6eSMn2GUNMI-_gY/view?usp=sharing
[uniformers32ssv2]: https://drive.google.com/file/d/1-5Q5S-NRmHFkIBygaRYpJLT7aI9cpmky/view?usp=sharing
[uniformerb16ssv2]: https://drive.google.com/file/d/1s78JBpnykyJ7pYEGRDK8BZL9seIRIUlC/view?usp=sharing
[uniformerb32ssv2]: https://drive.google.com/file/d/1-rpMARXnyvyj6YUJkIvVqtna86egpjoS/view?usp=sharing