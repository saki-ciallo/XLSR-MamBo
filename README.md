# XLSR-MamBo Official Implementation

This repository contains the official code and pretrained models for the paper **["XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection"](https://arxiv.org/abs/2601.02944)**, accepted at **[ACL 2026 Findings](https://2026.aclweb.org/)**.




## Downloads
**Models & Checkpoints:** The trained models and checkpoints (corresponding to the models in Table 3) are available on [Google Drive](https://drive.google.com/drive/folders/108a-gFEGhQU2_p16MvD2Rk7qeSyHEBpw?usp=sharing).

**Extended Results:** Additional results can be accessed via [Baidu Drive](https://pan.baidu.com/s/1P1qLazEr0URd280BBsCohw?pwd=ks4e) (Password: `ks4e`).



## Getting Started
This repository requires two specific folders to be placed in the root directory before running:
1. `./2021`: The official [ASVspoof 2021 baseline repository](https://github.com/asvspoof-challenge/2021).
2. `./keys`: The evaluation keys for [LA](https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz) and [DF](https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz) tasks, which can also be found in the 2021 repository.

> **Note:** For convenience, we have also uploaded these required folders and keys to our [Google Drive](https://drive.google.com/drive/folders/108a-gFEGhQU2_p16MvD2Rk7qeSyHEBpw?usp=sharing).

### Setup Environment
Our implementation relies on [Flash Attention](https://github.com/dao-ailab/flash-attention), [Mamba](https://github.com/state-spaces/mamba), and [Hydra](https://github.com/goombalab/hydra).

Base Environment:
```
cuda==12.8
python==3.12
torch==2.9.0
torchaudio==2.9.0
torchvision==0.24.0
```

We recommend using Docker to set up the environment, though Conda is also a valid option. You can install the necessary dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Datasets
* Training: [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336)
* Evaluation: ASVspoof 2021 [DF](https://zenodo.org/record/4837263) / [LA](https://zenodo.org/record/4835108), [In-the-Wild](https://deepfake-total.com/in_the_wild), and [DFADD](https://huggingface.co/datasets/isjwdu/DFADD)

### Pre-trained XLSR

We utilize the [XLSR-300M](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_XLSR_300M.html) model provided by `torchaudio` as our pre-trained backbone.

## RawBoost
To improve computational efficiency and ensure support for complex numbers, we optimized the implementation of the `SSI_additive_noise` function from the original RawBoost repository. You can find the detailed inline comments in our source code.


## Usage
We provide a comprehensive bash script, `train_multiple.sh`, which contains the commands for training, testing, and scoring. Please refer to this file for execution details.

## Results
The table below reports the Equal Error Rate (EER %). Note that D1-D3 and F1-F2 are subsets of the DFADD dataset.

<div align="center">
  
|Model|21LA|21DF|ITW|D1|D2|D3|F1|F2|
|-|-|-|-|-|-|-|-|-|
|MamBo-1-Mamba2-N2 |0.79|2.01|5.57|1.69|1.69|0.00|9.00 |12.54|
|MamBo-2-Hydra-N1  |0.80|1.84|6.24|1.84|1.69|0.00|5.32 |8.85 | 
|MamBo-3-Hydra-N3  |0.81|1.70|4.97|1.84|1.33|0.00|11.36|16.01|
|MamBo-4-Hydra-N1  |0.98|1.43|5.17|1.33|1.84|0.00|14.17|19.34|

</div>

## Citation
If you find our work or this repository useful for your research, please consider citing our paper:

```
@article{ng2026xlsr,
  title={XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection},
  author={Ng, Kwok-Ho and Song, Tingting and WU, Yongdong and Xia, Zhihua},
  journal={arXiv preprint arXiv:2601.02944},
  year={2026}
}
```

## Acknowledgements
This work is built upon and references the following open-source projects:

1. [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing)
2. [RawBMamba](https://github.com/cyjie429/RawBMamba)
3. [XLSR-Mamba](https://github.com/swagshaw/XLSR-Mamba)
4. [Fake-Mamba](https://github.com/xuanxixi/Fake-Mamba)

## License
All code and models in this repository are licensed under the [MIT License](https://github.com/saki-ciallo/XLSR-MamBo/blob/main/LICENSE).

## Contact Us
If you are interested in our work or have any questions regarding this repository, please feel free to open an issue or reach out via [email](mailto:kwokhong@stu2024.jnu.edu.cn).
