import os
import torch
import numpy as np
import polars as pl
from collections import Counter
from RawBoost import process_Rawboost_feature
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from glob import glob
import warnings
import logging
import librosa
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
logging.getLogger("audioread").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

LABEL_MAP = {'spoof': 0, 'bonafide': 1}

def dataset_baseinfo(input_dataset, split_name: str):
    label_list = input_dataset.protocol_data['label_encoded'].tolist()
    label_counts = Counter(label_list)
    count_0 = label_counts.get(0, 0) # 标签 0 的数量 (Bonafide)
    count_1 = label_counts.get(1, 0) # 标签 1 的数量 (Spoof)
    print(f"--- {split_name} 标签数量统计 ---")
    print(f"标签 0 (Bonafide) 数量: {count_0}")
    print(f"标签 1 (Spoof) 数量: {count_1}")
    print(f"总样本数量: {len(label_list)}")


def pad_wave_repeat(waveform: np.ndarray, max_len: int) -> np.ndarray:
    """对音频进行重复填充或截断至固定长度"""
    waveform_len = len(waveform)
    if waveform_len <= 0:
        raise ValueError("Waveform length must be positive")
    elif waveform_len == max_len:
        return waveform
    elif waveform_len > max_len:
        return waveform[:max_len]
    elif waveform_len < max_len:
        repeats = -(-max_len // waveform_len)
        # repeats = int(np.ceil(max_len / waveform_len))
        padded = np.tile(waveform, repeats)[:max_len]
        return padded
    else:
        raise ValueError("Unexpected condition in pad_wave_repeat")

class DFADD_Dataset(Dataset):
    """
    支持 train/valid/test 三个子集的 DFADD 数据集。
    通过 protocol 文件加载样本列表，并根据标签从不同目录加载音频文件
    """
    def __init__(self,
                types: str,
                spoof_data_dir: str,           # e.g., "DFADD/DATASET_GradTTS"
                bonafide_data_dir: str,        # e.g., "DFADD/DATASET_VCTK_BONAFIDE"
                fixed_length_samples: int,
                args_main = None
                ):
        self.types = types
        self.spoof_dir = os.path.join(spoof_data_dir, types) # DFADD/DATASET_GradTTS/train
        self.bonafide_dir = os.path.join(bonafide_data_dir, types) # DFADD/DATASET_VCTK_BONAFIDE/train
        self.protocol_file_path = os.path.join(spoof_data_dir, f"{types}.txt") #DFADD/DATASET_GradTTS/train.txt
        self.fixed_length_samples = fixed_length_samples
        self.args_main = args_main
        
        if not os.path.exists(self.protocol_file_path):
            raise FileNotFoundError(f"Protocol not found: {self.protocol_file_path}")

        data = pl.read_csv(
            self.protocol_file_path,
            separator=' ',
            has_header=False,
            new_columns=['speaker', 'key', 'col3', 'col4', 'label'],
            truncate_ragged_lines=True
        )

        self.file_ids = data['key'].to_list() # 'p230_409' 或 'p230_409_GradTTS'
        self.labels = data['label'].replace_strict(LABEL_MAP).to_list() # 标签化 'bonafide' 1 or 'spoof' 0
        
    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        key = self.file_ids[idx]
        label = self.labels[idx]

        if label == 1:
            file_path = os.path.join(self.bonafide_dir, f"{key}.wav") # 真实文件均为wav格式
        else:
            file_path = os.path.join(self.spoof_dir, f"{key}.flac") # 合成文件均为flac格式

        waveform, fs = librosa.load(file_path, sr=16000)
        if self.args_main.algo != 0 and self.types in {'train', 'valid'}:
            waveform = process_Rawboost_feature(waveform, fs, self.args_main, self.args_main.algo)
        waveform = pad_wave_repeat(waveform, max_len=self.fixed_length_samples)
        waveform = torch.from_numpy(waveform).float()

        return waveform, torch.tensor(label, dtype=torch.long)


class ASVspoof2019LADataset(Dataset):
    def __init__(self,
                 types: str,
                 audio_files_dir: str,
                 protocol_file_path: str,
                 fixed_length_samples: int,
                 args_main = None
                 ):
        self.types = types
        self.audio_files_dir = audio_files_dir
        self.protocol_file_path = protocol_file_path
        self.fixed_length_samples = fixed_length_samples
        self.args_main = args_main
        
        if not os.path.exists(self.protocol_file_path):
            raise FileNotFoundError(f"Protocol not found: {self.protocol_file_path}")
        
        self.protocol_data = pl.read_csv(
            protocol_file_path, 
            separator=' ', 
            has_header=False,
        )
        
        self.file_ids = self.protocol_data.get_column('column_2').to_list()
        self.labels = self.protocol_data.get_column('column_5').replace_strict(LABEL_MAP).to_list()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        label = self.labels[index]
        
        file_path = os.path.join(self.audio_files_dir, f"{file_id}.flac")
        
        waveform, fs = librosa.load(file_path, sr=16000)
        if self.args_main.algo != 0 and self.types in {'train', 'dev'}:
            waveform = process_Rawboost_feature(waveform, fs, self.args_main, self.args_main.algo)
        waveform = pad_wave_repeat(waveform, max_len=self.fixed_length_samples)
        waveform = torch.from_numpy(waveform).float()
        
        return waveform, torch.tensor(label, dtype=torch.long)


def asv19_dataloader(types: str, batch_size: int = 16, num_workers: int = 0, args_main = None):
    audio_files_dir = f'../Datasets/ASVspoof2019/ASVspoof2019_LA_{types}/flac'
    if types == 'train':
        protocol_file_path = f'../Datasets/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    else:
        protocol_file_path = f'../Datasets/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{types}.trl.txt'

    dataset = ASVspoof2019LADataset(
        types,
        audio_files_dir,
        protocol_file_path,
        fixed_length_samples = args_main.fixed_length_samples,
        args_main=args_main,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = True if types == 'train' else False,
        drop_last=True if types == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    return loader


def dfadd_dataloader(types: str, subset: str, batch_size: int = 16, num_workers: int = 0, args_main = None):
    
    dataset = DFADD_Dataset(
        spoof_data_dir=f'../Datasets/DFADD/DATASET_{subset}',
        bonafide_data_dir='../Datasets/DFADD/DATASET_VCTK_BONAFIDE',
        types=types, # train/valid/test
        fixed_length_samples = args_main.fixed_length_samples,
        args_main=args_main,
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = True if types == 'train' else False,
        drop_last=True if types == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    return loader

