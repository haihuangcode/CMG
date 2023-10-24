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
git clone https://github.com/haihuangcode/ACMG-MUR
cd ACMG-MUR
# You don't actually have to install all the libraries in the txt file, you can choose to install them as needed.
pip install -r requirements.txt
```

- ##### Pretrain
```python
cd ACMG-MUR/code/src
./pretrain.sh
```

- ##### AVE_downstream
```python
cd ACMG-MUR/code/src
./ave.sh
```

- ##### AVVP_downstream
```python
cd ACMG-MUR/code/src
./avvp.sh
```

- ##### AVE_AVVP_downstream
```python
cd ACMG-MUR/code/src
./ave_avvp.sh
```

- ##### UCF_VGGSOUND_downstream
```python
cd ACMG-MUR/code/src
./ucf_vggsound.sh
```

- ##### AVS_downstream
```python
cd ACMG-MUR/code/AVSBench_downstream/avs_scripts/avs_s4
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

## ✏Model Checkpoints

will be soon

## ✏Data feature
| Feature      | Link                                                 |
| -------      | -----------------------------------------------------|
| VGGSOUND     | will be soon|
| AVE          | will be soon|
| AVVP         | will be soon|
| AVE_AVVP     | will be soon|
| UCF_VGGSOUND | will be soon|
| AVS          | will be soon|

For the video and audio feature extraction method, please refer to [AVE](https://github.com/YapengTian/AVE-ECCV18), text is based on the label to generate a description-focused statement of approximately 10 words in length.