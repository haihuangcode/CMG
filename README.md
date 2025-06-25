# Achieving Cross Modal Generalization with Multimodal Unified Representation, NeurIPS 2023



![model](figs/model.png)

This is the Pytorch implementation of our paper:

Achieving Cross Modal Generalization with Multimodal Unified Representation

[Yan Xia](https://scholar.google.com/citations?user=6kEbV3IAAAAJ&hl), [Hai Huang](https://scholar.google.com/citations?user=FKvBzQwAAAAJ), [Jieming Zhu](https://scholar.google.com/citations?user=oNKerP8AAAAJ), [Zhou Zhao](https://scholar.google.com.hk/citations?user=IIoFY90AAAAJ)

In NeurIPS 2023

------

### 📝Requirements and Installation

- ##### Getting Started
**Due to the version conflict between bert_embedding's dependency on NumPy and other libraries, directly installing according to requirements.txt may cause issues. For more details, you can refer to this [issue](https://github.com/haihuangcode/CMG/issues/14)."**
```python
git clone https://github.com/haihuangcode/CMG
cd CMG
# You don't actually have to install all the libraries in the txt file, you can choose to install them as needed.
# It is recommended to use Python 3.7, as some libraries used do not support higher versions of Python.
conda create -n your_env_name python=3.7
pip install -r requirements.txt
```

- ##### Pretrain
```python
# Before you begin pretraining, please make sure to modify the file paths under `args.dataset_name == 'vggsound_AVT'` in `pretrain.py` to your own paths.
# Additionally, update the `file_path` and `self.label2prompt = pd.read_csv('')` paths in `dataset/VGGSOUND_dataset.py`.
# The model save path is located under `--model_save_path` in `configs/opts.py`.
# Please also remember to modify the paths related to downstream tasks and the corresponding dataset paths to your own paths.
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
@article{xia2024achieving,
  title={Achieving Cross Modal Generalization with Multimodal Unified Representation},
  author={Xia, Yan and Huang, Hai and Zhu, Jieming and Zhao, Zhou},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## ✏Model Checkpoints And Date Feature
You can choose to download from either Baidu Netdisk or Google Drive.

### Baidu Netdisk (If you are unable to access Google Drive, please use the following two links)
[data](https://pan.baidu.com/s/1CTcjMHVeG-8uo4HPWNNL9Q ) (pwd: 1234)
- 2023.11.07 Update https://github.com/haihuangcode/CMG/issues/1

[patch](https://pan.baidu.com/s/1rjVmRMut39kezw0FDZ7MwQ) (pwd: 1234)
- 2024.12.27 This is a patch for the previous data errors. Please download the complete data from the above and replace the csv files in the patch with the ones in `data/vggsound40k/data`, specifically replacing `vggsound-avel40k.csv` and `video_name_vggsound40k_checked.csv`. The previous https://github.com/haihuangcode/CMG/issues/13 regarding unsatisfactory model training results were caused by the incomplete csv files that were uploaded earlier, which only contained 20k data entries. I apologize for not noticing this earlier /(ㄒoㄒ)/~~

### Google Drive (Includes the complete data and patch)
[data+patch](https://drive.google.com/drive/folders/1ThGAXoqay7RanGwHz21qZGMEjF3W1VwS?usp=drive_link)

## ✏Directory

```
CMG
├── checkpoint
├── cnt.pkl
├── code
├── data
├── figs
├── paper
├── README.md
└── requirements.txt
```

## ✏Note
- For the video and audio feature extraction method, please refer to [AVE](https://github.com/YapengTian/AVE-ECCV18), text is based on the label to generate a description-focused statement of approximately 10 words in length.
- There is no validation set for the pre-training process, in this paper it is done by testing the performance of each model on the downstream of the [AVE](https://github.com/YapengTian/AVE-ECCV18), and the model with the best performance tests the rest of the downstream tasks, so the [AVE](https://github.com/YapengTian/AVE-ECCV18) can be regarded as a validation set and the model with the best pre-training appears in the first 5 epochs.
- Pretraining can be performed using just one GPU, such as 4090 or A100. The experimental results in the paper were obtained by running on 4090 or A100. Multi-GPU parallel training yielded poorer model performance, possibly due to issues between the mutual information minimization design in DCID and Pytorch (but this was an early experimental observation, and was not re-verified after the code was finalized, since single GPU pretraining was sufficient).

## 👍Acknowledgments

Our code is based on [AVE](https://github.com/YapengTian/AVE-ECCV18), [AVVP](https://github.com/YapengTian/AVVP-ECCV20), [PSP](https://github.com/jasongief/PSP_CVPR_2021), [CPSP](https://github.com/jasongief/CPSP), [VGGSOUND](https://github.com/hche11/VGGSound), [AVS](https://github.com/OpenNLPLab/AVSBench).
