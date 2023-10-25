# Achieving Cross Modal Generalization with Multimodal Unified Representation, NeurIPS 2023



![model](figs/model.png)

This is the Pytorch implementation of our paper:

Achieving Cross Modal Generalization with Multimodal Unified Representation

[Yan Xia](https://scholar.google.com/citations?user=6kEbV3IAAAAJ&hl), [Hai Huang](https://github.com/haihuangcode), [Jieming Zhu](https://scholar.google.com/citations?user=oNKerP8AAAAJ), [Zhou Zhao](https://scholar.google.com.hk/citations?user=IIoFY90AAAAJ)

In NeurIPS 2023

------

### 📝Requirements and Installation

- ##### Getting Started

```python
git clone https://github.com/haihuangcode/CMG
cd CMG
# You don't actually have to install all the libraries in the txt file, you can choose to install them as needed.
pip install -r requirements.txt
```

- ##### Pretrain
```python
cd CMG/code/src
./pretrain.sh
```

- ##### AVE_downstream
```python
cd CMG/code/src
./ave.sh
```

- ##### AVVP_downstream
```python
cd CMG/code/src
./avvp.sh
```

- ##### AVE_AVVP_downstream
```python
cd CMG/code/src
./ave_avvp.sh
```

- ##### UCF_VGGSOUND_downstream
```python
cd CMG/code/src
./ucf_vggsound.sh
```

- ##### AVS_downstream
```python
cd CMG/code/AVSBench_downstream/avs_scripts/avs_s4
./train.sh
./test.sh
```

## 🎓Cite

If you find this work useful, please consider citing it.

```
cite of ACMG-MUR
```

## 👍Acknowledgments

Our code is based on [AVE](https://github.com/YapengTian/AVE-ECCV18), [AVVP](https://github.com/YapengTian/AVVP-ECCV20), [PSP](https://github.com/jasongief/PSP_CVPR_2021), [CPSP](https://github.com/jasongief/CPSP), [VGGSOUND](https://github.com/hche11/VGGSound), [AVS](https://github.com/OpenNLPLab/AVSBench).

## ✏Model Checkpoints And Date Feature

[Baidu Disk](https://pan.baidu.com/s/1u6gNTyclDSO5e1ONOKwqkA ) (pwd: 1234)

### Note
- For the video and audio feature extraction method, please refer to [AVE](https://github.com/YapengTian/AVE-ECCV18), text is based on the label to generate a description-focused statement of approximately 10 words in length.
- There is no validation set for the pre-training process, in this paper it is done by testing the performance of each model on the downstream of the [AVE](https://github.com/YapengTian/AVE-ECCV18), and the model with the best performance tests the rest of the downstream tasks, so the [AVE](https://github.com/YapengTian/AVE-ECCV18) can be regarded as a validation set and the model with the best pre-training appears in the first 5 epochs.

## ✏Directory

```
CMG
├── checkpoint
├── cnt.pkl
├── code
├── data
├── figs
├── README.md
└── requirements.txt
```