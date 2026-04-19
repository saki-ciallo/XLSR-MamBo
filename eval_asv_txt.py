import os
import torch
import random
import glob
import re
import numpy as np
from tqdm import tqdm
import polars as pl
from torch.utils.data import Dataset, DataLoader
from importlib import import_module
import argparse
import gc
import warnings
import logging
import librosa
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
logging.getLogger("audioread").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_safe(model, path, device):
    """安全加载模型权重，兼容torch.compile包装后保存的state_dict"""
    state_dict = torch.load(path, map_location=device)
    
    # 检查是否是torch.compile包装后保存的（key带有_orig_mod前缀）
    has_orig_mod_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    
    if has_orig_mod_prefix:
        # 移除_orig_mod.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[len('_orig_mod.'):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        print("Detected torch.compile wrapped state_dict, removed '_orig_mod.' prefix")
    
    model.load_state_dict(state_dict)
    return model

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def reproducibility(random_seed):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return


def pad_wave_repeat(waveform, max_len):
    """对音频进行重复填充或截断至固定长度"""
    waveform_len = len(waveform)
    if waveform_len <= 0:
        raise ValueError("Waveform length must be positive")
    elif waveform_len == max_len:
        # 如果长度相等，则直接返回
        return waveform
    elif waveform_len > max_len:
        return waveform[:max_len]
    elif waveform_len < max_len:
        # 重复填充
        repeats = -(-max_len // waveform_len)
        # repeats = int(np.ceil(max_len / waveform_len))
        padded = np.tile(waveform, repeats)[:max_len]
        return padded
    else:
        raise ValueError("Unexpected condition in pad_wave_repeat")


def load_files_asv19(protocol_file_path):
    """加载指定目录下的 .flac 文件并进行预处理
    data_dir= "../Datasets/ASVspoof2019/ASVspoof2019_LA_eval/flac/"
    protocol_file_path = '../Datasets/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    LA_0031 LA_E_5932896 - A13 spoof
    LA_0030 LA_E_5849185 - - bonafide
    如果 'system_id' 为 '-'，设置为A00
    """
    # faster than pandas and openfile
    protocol_data = pl.read_csv(
        protocol_file_path, 
        separator=' ', 
        has_header=False,
        )
    # 1: speaker_id, 2: file_id, 3: unused, 4: system_id, 5: label
    file_ids = protocol_data.get_column('column_2').to_list()
    system_ids = protocol_data.get_column('column_4').to_list()
    labels = protocol_data.get_column('column_5').to_list()

    print(f"Requested {len(file_ids)} files for processing")
    return file_ids, system_ids, labels


def load_files_asv21(protocol_file_path):
    """
    从asv21df/la协议文件中加载文件ID、系统ID和标签
    21DF格式: LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -
    21LA格式: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    """
    # faster than pandas and openfile
    protocol_data = pl.read_csv(
        protocol_file_path,
        separator=' ',
        has_header=False,
        truncate_ragged_lines=True,  # 处理不等长行
    )

    # 提取需要的列 (索引从0开始)
    # 2: file_id, 5: system_id, 6: label
    file_ids = protocol_data.get_column('column_2').to_list()
    system_ids = protocol_data.get_column('column_5').to_list() # 格式兼容
    labels = protocol_data.get_column('column_6').to_list()

    print(f"Loaded {len(file_ids)} files from {protocol_file_path}")
    return file_ids, system_ids, labels


def load_files_itw(protocol_file_path):
    """
    从in-the-wild协议文件中加载文件ID和标签
    Datasets/in_the_wild/meta.csv
    格式：
    file,speaker,label
    0.wav,Alec Guinness,spoof
    """
    # faster than pandas and openfile
    protocol_data = pl.read_csv(
        protocol_file_path,
        separator=',',
        has_header=True,
        truncate_ragged_lines=True,
    )

    file_ids = protocol_data['file'].to_list()
    system_ids = protocol_data['speaker'].to_list() # 格式兼容
    labels = protocol_data['label'].to_list()

    print(f"Loaded {len(file_ids)} files from {protocol_file_path}")
    return file_ids, system_ids, labels


def load_files_dfadd(protocol_file_path):
    protocol_data = pl.read_csv(
        protocol_file_path,
        separator=' ',
        has_header=False,
        new_columns=['speaker', 'key', 'col3', 'col4', 'label'],
        truncate_ragged_lines=True,
    )

    file_ids = protocol_data['key'].to_list()
    system_ids = protocol_data['speaker'].to_list() # 格式兼容
    labels = protocol_data['label'].to_list()

    print(f"Loaded {len(file_ids)} files from {protocol_file_path}")
    return file_ids, system_ids, labels



class ASVspoof_Dataset_Eval(Dataset):
    """
    兼容 ASVspoof2019 和 ASVspoof2021 的评估数据集加载
    """
    def __init__(self, file_ids, system_ids, labels, data_dir, fixed_length_samples):
        self.file_ids = file_ids
        self.system_ids = system_ids
        self.labels = labels
        self.data_dir = data_dir
        self.fixed_length_samples = fixed_length_samples
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        wav_id = self.file_ids[idx]
        file_path = os.path.join(self.data_dir, f"{wav_id}.flac")
        waveform, _ = librosa.load(file_path, sr=16000)
        padded_y = pad_wave_repeat(waveform, max_len=self.fixed_length_samples)
        final_waveform = torch.from_numpy(padded_y).float()
        
        return {
            'wav_id': wav_id,
            'waveform': final_waveform,
            'system_id': self.system_ids[idx],
            'label': self.labels[idx]
        }

class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, file_ids, system_ids, labels, data_dir, fixed_length_samples):
        self.file_ids = file_ids
        self.system_ids = system_ids
        self.labels = labels
        self.data_dir = data_dir
        self.fixed_length_samples = fixed_length_samples

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        wav_id = self.file_ids[idx]
        label = self.labels[idx]
        
        file_path = os.path.join(self.data_dir, f"{wav_id}") # 文件名已包含扩展名 "0.wav"
        waveform, _ = librosa.load(file_path, sr=16000)
        padded_y = pad_wave_repeat(waveform, max_len=self.fixed_length_samples)
        final_waveform = torch.from_numpy(padded_y).float()
        
        return {
            'wav_id': wav_id,
            'waveform': final_waveform,
            'system_id': self.system_ids[idx],
            'label': self.labels[idx]
        }


class Dataset_DFADD_eval(Dataset):
    def __init__(self, file_ids, system_ids, labels, data_dir, fixed_length_samples):
        self.file_ids = file_ids
        self.system_ids = system_ids
        self.labels = labels
        self.spoof_dir = data_dir
        self.bonafide_dir = "../Datasets/DFADD/DATASET_VCTK_BONAFIDE/test/"
        self.fixed_length_samples = fixed_length_samples

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        wav_id = self.file_ids[idx]
        label = self.labels[idx]
        if label == "bonafide":
            file_path = os.path.join(self.bonafide_dir, f"{wav_id}.wav") # 真实文件均为wav格式
        else:
            file_path = os.path.join(self.spoof_dir, f"{wav_id}.flac") # 合成文件均为flac格式
        
        waveform, _ = librosa.load(file_path, sr=16000)
        padded_y = pad_wave_repeat(waveform, max_len=self.fixed_length_samples)
        final_waveform = torch.from_numpy(padded_y).float()
        
        return {
            'wav_id': wav_id,
            'waveform': final_waveform,
            'system_id': self.system_ids[idx],
            'label': self.labels[idx]
        }


        
def asv_evals(model, device, task_out_fold, save_name, dataset_name, fixed_length_samples):
    """
    测试函数，用于对 ASVspoof2019 数据集进行评估并打分
    dataset_name: 'asv19', 'asv21df', 'asv21la', 'itw'
    """
    # 测试集位置
    if dataset_name == 'asv19':
        data_dir = "../Datasets/ASVspoof2019/ASVspoof2019_LA_eval/flac/"
        protocol_file_path = '../Datasets/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    elif dataset_name == 'asv21df':
        data_dir = "../Datasets/ASVspoof2021/ASVspoof2021_DF_eval/flac/"
        protocol_file_path = '../Datasets/ASVspoof2021/ASVspoof2021_DF_eval/keys/DF/CM/trial_metadata.txt'
    elif dataset_name == 'asv21la':
        data_dir = "../Datasets/ASVspoof2021/ASVspoof2021_LA_eval/flac/"
        protocol_file_path = '../Datasets/ASVspoof2021/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt'
    elif dataset_name == 'itw':
        data_dir = "../Datasets/in_the_wild/"
        protocol_file_path = '../Datasets/in_the_wild/meta.csv'
    elif dataset_name == 'D1':
        data_dir = "../Datasets/DFADD/DATASET_GradTTS/test"
        protocol_file_path = '../Datasets/DFADD/DATASET_GradTTS/test.txt'
    elif dataset_name == 'D2':
        data_dir = "../Datasets/DFADD/DATASET_NaturalSpeech2/test"
        protocol_file_path = '../Datasets/DFADD/DATASET_NaturalSpeech2/test.txt'
    elif dataset_name == 'D3':
        data_dir = "../Datasets/DFADD/DATASET_StyleTTS2/test"
        protocol_file_path = '../Datasets/DFADD/DATASET_StyleTTS2/test.txt'
    elif dataset_name == 'F1':
        data_dir = "../Datasets/DFADD/DATASET_MatchaTTS/test"
        protocol_file_path = '../Datasets/DFADD/DATASET_MatchaTTS/test.txt'
    elif dataset_name == 'F2':
        data_dir = "../Datasets/DFADD/DATASET_PflowTTS/test"
        protocol_file_path = '../Datasets/DFADD/DATASET_PflowTTS/test.txt'
    
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        
    # 根据不同数据集设置输出文件路径，用dataset_name区分
    output_file = os.path.join(task_out_fold, dataset_name, save_name) # 同路径放置
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    model.eval()
    
    # 加载音频文件
    if dataset_name == "asv19":
        file_ids, system_ids, labels = load_files_asv19(protocol_file_path)
    elif dataset_name in ["asv21df", "asv21la"]:
        file_ids, system_ids, labels = load_files_asv21(protocol_file_path)
    elif dataset_name == "itw":
        file_ids, system_ids, labels = load_files_itw(protocol_file_path)
    elif dataset_name in ['D1','D2','D3','F1','F2']:
        file_ids, system_ids, labels = load_files_dfadd(protocol_file_path)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    if dataset_name == "itw":
        dataset = Dataset_in_the_wild_eval(file_ids, system_ids, labels, data_dir, fixed_length_samples)
    elif dataset_name in ['D1','D2','D3','F1','F2']:
        dataset = Dataset_DFADD_eval(file_ids, system_ids, labels, data_dir, fixed_length_samples)
    else:
        dataset = ASVspoof_Dataset_Eval(file_ids, system_ids, labels, data_dir, fixed_length_samples)
        
    # 根据硬件调整 batch_size 和 num_workers，推理时通常可以用更大的 batch_size
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=12,
                            drop_last=False,
                            pin_memory=True,
                            multiprocessing_context='spawn')
    
    wave_list, score_list = [], []
    all_system_ids = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            # Correct 正确从 dict 取数据
            batch_x = batch['waveform'].to(device)      # [32, 66800]
            utt_id  = batch['wav_id']                   # list of 32 str
            
            # float32
            # batch_out = model(batch_x)
            # bfloat16，会带来分数下降，但是速度提升一倍
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                batch_out = model(batch_x)

            # 标准方法    
            batch_score = (batch_out[:, 1] - batch_out[:, 0]).cpu().ravel()
            # 其他方法
            # batch_score = batch_out[:, 1].cpu().ravel() # 分数与bf16结果别无二致，相比较标准方法均差一些

            wave_list.extend(utt_id)
            all_system_ids.extend(batch['system_id'])
            all_labels.extend(batch['label'])
            score_list.extend(batch_score)
    if dataset_name == 'asv19':
    # 保存结果到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as fh:
            for file_id, sys_id, label, score in zip(wave_list, all_system_ids, all_labels, score_list):
                fh.write(f'{file_id} {sys_id} {label} {score}\n')
    else:
        # 保存结果到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as fh:
            for file_id, score in zip(wave_list, score_list):
                fh.write(f'{file_id} {score}\n')

    print(f'Eval result saved to {output_file}')

    
def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        current = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current)
        print(f"Force set to GPU {current}: {name}")
        return torch.device(f'cuda:{current}')
    else:
        raise RuntimeError("没有找到 CUDA 设备。\n")

def main():
    """
    路径使用 train_config.py 中的设置，信息复用以简化代码
    """
    parser = argparse.ArgumentParser(description='Evaluate ASVspoof2019 model')
    parser.add_argument('--epochs', type=int, default=20, metavar='N') # 兼容用
    parser.add_argument("--dataset", type=str, help="dataset name", required=True, default='asv19')
    parser.add_argument("--task", type=str, required=True, help="The SSM four types: xlsr_Mamba1, xlsr_Mamba2, xlsr_Hydra, xlsr_GDN")
    parser.add_argument('--fixed-length-samples', type=int, default=66800, help='Fixed length of audio samples')
    
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba2', help='Only Mamba2 / Hydra / GDN block types are supported')
    parser.add_argument('--block-layers', type=int, required=True, help='The number of block layers')
    parser.add_argument('--block-sequence', type=str, required=True, help='The block sequence')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    
    parser.add_argument('--gpu-id', type=int, default=0, metavar='N')
    
    parser.add_argument("--out-fold", type=str, help="output folder", required=True, default='./outputs/')
    
    parser.add_argument('--seed', type=int, default=114514, metavar='S')
    
    parser.add_argument('--algo', type=int, default=0, required=True, 
                    help='Rawboost algos discriptions. Follow two diff. setting (3 for DF, 5 for LA and ITW) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--freeze', action="store_true", help='Whether to fine-tune the XLS-R or not')
    
    parser.add_argument('--force', action='store_true', default=False,
                    help='Force re-evaluation even if result file exists')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')

    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    
    reproducibility(args.seed)
    
    device = get_device(gpu_id=args.gpu_id)
    print(f"Using device: {device}")
    
    config_module = import_module('train_config')
    task_config = config_module.TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Unknown task: {args.task}. Please define it in train_config.py")
    model = task_config.model_classifier(task_config.model_config,
                                        freeze=args.freeze,
                                        block_layers=args.block_layers,
                                        block_sequence=args.block_sequence,
                                        n_mamba=args.n_mamba,
                                        block_cls=args.block_cls,
                                        fixed_length_samples=args.fixed_length_samples
                                        ).to(device)
    
    fixed_length_samples = args.fixed_length_samples
    
    # 创建目录，直接使用训练所用路径 
    b_repeat = args.block_layers
    b_sequence = args.block_sequence
    if "n_" in b_sequence:
        b_sequence = b_sequence.replace("n_", str(args.n_mamba))
    mixer_type = args.block_cls
    fz_lb = "freeze" if args.freeze else "finetune"
    dfadd_subset = "" if args.dfadd_subset == "None" else f"_{args.dfadd_subset}" # asv19 不需要标注
    # task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}{dfadd_subset}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence)
    # rebuttal 用
    task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}_{args.fixed_length_samples}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence) # 任务目录

    
    # args.dataset 用于指定使用什么数据集进行评估，包括 dfadd 的各个子集，使用文章中的D/F命名
    # args.dfadd_subset 用于指定使用某个 DFADD 子集训练的模型
    
    
    def eval_file_exists(save_name):
        eval_file_path = os.path.join(task_out_fold, args.dataset, save_name)
        return os.path.exists(eval_file_path)
    def extract_epoch(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"best_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1
    
    # 保留基础模型引用，用于加载权重
    base_model = model
    compiled_model = None  # 编译后的模型，只编译一次
    
    # 1. 对保存的损失最低模型进行评估
    best_eval_name = "eval_model_best.txt"
    if not args.force and eval_file_exists(best_eval_name):
        print(f"Skipping: {best_eval_name} already exists for dataset '{args.dataset}'")
    else:
        best_model_path = os.path.join(task_out_fold, "model_best.pt")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found: {best_model_path}")
        # 安全加载（兼容torch.compile包装后保存的旧模型）
        base_model = load_model_safe(model, best_model_path, device)
        
        compiled_model = torch.compile(base_model, mode="max-autotune-no-cudagraphs")

        asv_evals(compiled_model, device, task_out_fold, save_name="eval_model_best.txt", dataset_name=args.dataset,
                  fixed_length_samples=fixed_length_samples)
                
    # 2. 检查是否存在checkpoint目录
    checkpoint_dir = os.path.join(task_out_fold, "checkpoint")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Info: Checkpoint directory not found, skipping evaluation of epoch models: {checkpoint_dir}")
    else:
        # 如果checkpoint目录存在，则加载各个epoch的模型，名称为 "model_epoch_{epoch}.pt"
        pattern = os.path.join(checkpoint_dir, "best_*.pt")
        checkpoint_files = glob.glob(pattern)
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir} with pattern 'best_*.pt'")
        else:
            checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
            for checkpoint_file in checkpoint_files:
                base_name = os.path.splitext(os.path.basename(checkpoint_file))[0] # 拆分文件名和扩展名
                
                # 跳过 best_0.pt，因为它与 model_best.pt 重叠
                if base_name == "best_0":
                    print(f"Skipping: {base_name}.pt (same as model_best.pt)")
                    continue
                
                eval_save_name = f"eval_{base_name}.txt"
                if not args.force and eval_file_exists(eval_save_name):
                    print(f"Skipping: {eval_save_name} already exists for dataset '{args.dataset}'")
                    continue

                # 加载权重到 base_model（compiled_model 共享相同的参数）
                base_model = load_model_safe(model, checkpoint_file, device)

                # 首次编译（如果之前跳过了 model_best.pt 的评估）
                if compiled_model is None:
                    compiled_model = torch.compile(base_model, mode="max-autotune-no-cudagraphs")
                
                # 使用编译后的模型进行评估（参数已通过 base_model 更新）
                asv_evals(compiled_model, device, task_out_fold, save_name=eval_save_name, dataset_name=args.dataset,fixed_length_samples=fixed_length_samples)
                        


if __name__ == '__main__':
    main()
