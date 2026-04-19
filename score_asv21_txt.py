import argparse
import os
import re
import glob
from importlib import import_module
from contextlib import redirect_stdout
import subprocess


def run_official_scoring(cm_score_file, track, subset="eval"):
    """
    调用官方 ./2021/eval-package/main.py 进行评分
    """
    # example:
    # python ./2021/eval-package/main.py --cm-score-file ./outputs/<...>/eval_model_best.txt --track LA --subset eval
    cmd = [
        "python", "./2021/eval-package/main.py",
        "--cm-score-file", cm_score_file,
        "--track", track,
        "--subset", subset
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Scoring FAILED for {cm_score_file}")
        print(result.stderr)
        return None
    
    output = result.stdout
    print(output.strip())
    return output.strip()


def main():
    parser = argparse.ArgumentParser(description='Get ASVspoof2021 LA/DF EER and min-tDCF scores using official script')
    parser.add_argument("--dataset", type=str, help="dataset name", required=True, choices=['asv21la', 'asv21df'])
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument('--fixed-length-samples', type=int, default=66800, help='Fixed length of audio samples')
    
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba2', help='Only Mamba2 / Hydra / GDN block types are supported')
    parser.add_argument('--block-layers', type=int, required=True, help='Total number of blocks in the model')
    parser.add_argument('--block-sequence', type=str, required=True, help='block sequence for Mamba2 model, ref. class StackedMamba2')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    parser.add_argument("--out-fold", type=str, default='./outputs/', help="output folder")
    
    parser.add_argument('--algo', type=int, default=0, required=True, 
                help='Rawboost algos discriptions. Follow two diff. setting (3 for DF, 5 for LA) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                        5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--freeze', action="store_true", help='Whether to fine-tune the XLS-R or not')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')

    
    args = parser.parse_args()
    

    if args.dataset not in ['asv21la', 'asv21df']:
        raise ValueError("For this script, --dataset must be 'asv21la' or 'asv21df'")

    track = "LA" if args.dataset == "asv21la" else "DF"

    config_module = import_module('train_config')
    task_config = config_module.TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Unknown task: {args.task}. Please define it in train_config.py")
    b_repeat = args.block_layers
    b_sequence = args.block_sequence
    if "n_" in b_sequence:
        b_sequence = b_sequence.replace("n_", str(args.n_mamba))
    mixer_type = args.block_cls
    fz_lb = "freeze" if args.freeze else "finetune"
    task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence, args.dataset)
    # task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}_{args.fixed_length_samples}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence, args.dataset) # only for test

    eval_files = []
    
    # loss 最低模型结果
    best_eval_file = os.path.join(task_out_fold, "eval_model_best.txt")
    if os.path.exists(best_eval_file):
        eval_files.append(best_eval_file)
    else:
        print(f"Warning: Not found: {best_eval_file}")
        
    # 添加所有 epoch 模型结果
    epoch_pattern = os.path.join(task_out_fold, "eval_best_*.txt")
    epoch_files = glob.glob(epoch_pattern)
    if epoch_files:
        def extract_epoch(filepath):
            match = re.search(r"eval_best_(\d+)\.txt", os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        epoch_files = sorted(epoch_files, key=extract_epoch)
        eval_files.extend(epoch_files)
    else:
        print(f"Info: No 'eval_best_*.txt' files found.")

    if not eval_files:
        print(f"No evaluation result files found in {task_out_fold}, skipping scoring.")
    else:
        print(f"Found {len(eval_files)} evaluation result(s). Starting scoring...")
        for score_file in eval_files:
            base_name = os.path.splitext(os.path.basename(score_file))[0]
            output_txt = os.path.join(task_out_fold, f"score_{base_name}.txt")
            with open(output_txt, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    try:
                        run_official_scoring(score_file, track=track, subset="eval")
                    except Exception as e:
                        print(f"Error occurred while scoring {base_name}: {e}")
            print(f"Saved score result → {output_txt}")


    
    # 收集所有 score 文件的 min-tDCF 和 EER，生成 summarize.csv
    summarize_data = []
    
    # 根据数据集类型确定读取行号
    # asv21la: 第52行(index=51) min-tDCF，第94行(index=93) EER
    # asv21df: 只有第36行(index=35) EER，无 min-tDCF
    is_la = (args.dataset == "asv21la")
    
    def extract_metrics(lines):
        """从 score 文件中提取 min-tDCF 和 EER"""
        if is_la:
            min_tdcf = lines[51].strip().split()[-1] if len(lines) > 51 else "N/A"
            eer = lines[93].strip().split()[-1] if len(lines) > 93 else "N/A"
        else:  # asv21df
            min_tdcf = "N/A"  # DF 没有 min-tDCF
            eer = lines[35].strip().split()[-1] if len(lines) > 35 else "N/A"
        return min_tdcf, eer
    
    # 处理 score_eval_model_best.txt (level=0)
    best_score_file = os.path.join(task_out_fold, "score_eval_model_best.txt")
    if os.path.exists(best_score_file):
        try:
            with open(best_score_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            min_tdcf, eer = extract_metrics(lines)
            summarize_data.append((0, min_tdcf, eer))
        except Exception as e:
            print(f"Error reading {best_score_file}: {e}")
    
    # 处理 score_eval_best_*.txt (level=1,2,3,...)
    score_pattern = os.path.join(task_out_fold, "score_eval_best_*.txt")
    score_files = glob.glob(score_pattern)
    
    def extract_level(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"score_eval_best_(\d+)\.txt", filename)
        return int(match.group(1)) if match else -1
    
    score_files = sorted(score_files, key=extract_level)
    
    for score_file in score_files:
        level = extract_level(score_file)
        if level < 0:
            continue
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            min_tdcf, eer = extract_metrics(lines)
            summarize_data.append((level, min_tdcf, eer))
        except Exception as e:
            print(f"Error reading {score_file}: {e}")
    
    # 按 level 排序并写入 CSV（横向格式）
    if summarize_data:
        summarize_data.sort(key=lambda x: x[0])
        summarize_csv = os.path.join(task_out_fold, "summarize.csv")
        
        # 提取所有 level 和对应的值
        levels = [str(level) for level, _, _ in summarize_data]
        m_tdcf_values = [min_tdcf for _, min_tdcf, _ in summarize_data]
        eer_values = [eer for _, _, eer in summarize_data]
        
        with open(summarize_csv, 'w', encoding='utf-8') as f:
            # 第一行：level
            f.write("level," + ",".join(levels) + "\n")
            
            # 第二行：m-tDCF（LA 数据集才有）
            if is_la:
                f.write("m-tDCF," + ",".join(m_tdcf_values) + "\n")
            
            # 第三行：EER
            f.write("EER," + ",".join(eer_values) + "\n")
        
        print(f"Saved summary → {summarize_csv}")


if __name__ == '__main__':
    main()