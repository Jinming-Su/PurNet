# PurNet
Project for the TIP 2021 Paper "Salient Object Detection with Purificatory Mechanism and Structural Similarity Loss"

## Abstract
Image-based salient object detection has made great
progress over the past decades, especially after the revival of
deep neural networks. By the aid of attention mechanisms to
weight the image features adaptively, recent advanced deep
learning-based models encourage the predicted results to approximate
the ground-truth masks with as large predictable
areas as possible, thus achieving the state-of-the-art performance.
However, these methods do not pay enough attention to small
areas prone to misprediction. In this way, it is still tough to
accurately locate salient objects due to the existence of regions
with indistinguishable foreground and background and regions
with complex or fine structures. To address these problems, we
propose a novel convolutional neural network with purificatory
mechanism and structural similarity loss. Specifically, in order
to better locate preliminary salient objects, we first introduce
the promotion attention, which is based on spatial and channel
attention mechanisms to promote attention to salient regions.
Subsequently, for the purpose of restoring the indistinguishable
regions that can be regarded as error-prone regions of one model,
we propose the rectification attention, which is learned from the
areas of wrong prediction and guide the network to focus on
error-prone regions thus rectifying errors. Through these two
attentions, we use the Purificatory Mechanism to impose strict
weights with different regions of the whole salient objects and
purify results from hard-to-distinguish regions, thus accurately
predicting the locations and details of salient objects. In addition
to paying different attention to these hard-to-distinguish regions,
we also consider the structural constraints on complex regions
and propose the Structural Similarity Loss. The proposed loss
models the region-level pair-wise relationship between regions
to assist these regions to calibrate their own saliency values. In
experiments, the proposed purificatory mechanism and structural
similarity loss can both effectively improve the performance, and
the proposed approach outperforms 19 state-of-the-art methods
on six datasets with a notable margin. Also, the proposed method
is efficient and runs at over 27FPS on a single NVIDIA 1080Ti
GPU.

## Method
![Framework](https://github.com/Jinming-Su/PurNet/blob/master/assets/framework.png)
The framework of our approach. We first extract the common features by extractor, which provides the features for the other three subnetworks.
In detail, the promotion subnetwork produces promotion attention to guide the model to focus on salient regions, and the rectification subnetwork give the
rectification attention for rectifying the errors. These two kind of attentions are combined to formed the purificatory mechanism, which is integrated in the
purificatory subnetwork to refine the prediction of salient objects progressively.

## Quantitative Evaluation
![Quantitative Evaluation](https://github.com/Jinming-Su/PurNet/blob/master/assets/quantitative_evaluation.png)

## Qualitative Evaluation
![Qualitative Evaluation](https://github.com/Jinming-Su/PurNet/blob/master/assets/qualitative_evaluation.png)

## Usage
### Dataset
Download the DUTS dataset, and the corresponding superpixes can be downloaded. [BaiduYun](https://pan.baidu.com/s/1LSM5jgNapj-bpYDOzcVtYg) (Code: 2v1f)

### Training
```
1. install pytorch
2. train stage1, run python train.py
3. train stage2, run python train.py
4. train stage3, run python train.py
```
The trained checkpoint can be downloaded. [BaiduYun](https://pan.baidu.com/s/1-gvitX0mec3DPXMYShyHKw) (Code: c6sk)

### Testing
```
python test_code/test.py
```
The predicted saliency map of ECSSD can be downloaded. [BaiduYun](https://pan.baidu.com/s/1C95lorSyeKUaz05ZdWQ0mA) (Code: 1h4g) All results including ECSSD,
DUT-OMRON, PASCAL-S, HKU-IS, DUTS-TE, XPIE can all obtain by above testing code.

### Evaluation
```
matlab -nosplash -nodesktop -r evaluation_all
```

## Citation
```
@article{li2021salient,
  title={Salient object detection with purificatory mechanism and structural similarity loss},
  author={Li, Jia and Su, Jinming and Xia, Changqun and Ma, Mingcan and Tian, Yonghong},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={6855--6868},
  year={2021},
  publisher={IEEE}
}
```
