import argparse
import os
import re
import glob
from importlib import import_module
from score_asv19_txt import compute_eer
import polars as pl

def eval_to_score_file(score_file, cm_key_file):
    submission_scores = pl.read_csv(
        score_file,
        separator=' ',
        has_header=False,
        truncate_ragged_lines=True,
        new_columns=['file', 'score']  # 命名列
    )
    cm_data = pl.read_csv(
        cm_key_file,
        separator=' ',
        has_header=False,
        new_columns=['speaker', 'file', 'col3', 'col4', 'label'],
        truncate_ragged_lines=True
    )
    
    # 根据 file 名拼接，结果为 'file', 'score', 'label'
    cm_scores = submission_scores.join(
        cm_data, # 追加 cm_data
        on='file',
        how='inner'
    )
    
    # 提取 bona-fide 和 spoof 的分数（注意 score 是 float）
    bona_scores = cm_scores.filter(pl.col('label') == 'bonafide')['score'].to_numpy()
    spoof_scores = cm_scores.filter(pl.col('label') == 'spoof')['score'].to_numpy()
    
    eer_cm = compute_eer(bona_scores, spoof_scores)[0]
    # out_data = "eer: %.2f\n" % (100*eer_cm)
    # print(out_data)
    return eer_cm


def main():
    parser = argparse.ArgumentParser(description='Get DFADD EER scores using official script')
    parser.add_argument("--dataset", type=str, help="dataset name", required=True, choices=['D1', 'D2', 'D3', 'F1', 'F2'])
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument('--fixed-length-samples', type=int, default=66800, help='Fixed length of audio samples')
    
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba1', help='The SSM types: Mamba1, Mamba2, Hydra, GDN')
    parser.add_argument('--block-layers', type=int, required=True, help='Total number of blocks in the model')
    parser.add_argument('--block-sequence', type=str, required=True, help='block sequence for Mamba2 model, ref. class StackedMamba2')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    parser.add_argument("--out-fold", type=str, default='./outputs/', help="output folder")
    
    parser.add_argument('--algo', type=int, default=0, required=True, 
                help='Rawboost algos discriptions. Follow two diff. setting (3 for DF, 5 for LA and ITW) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                        5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--freeze', action="store_true", help='Whether to fine-tune the XLS-R or not')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')

    args = parser.parse_args()

    if args.dataset == 'D1':
        cm_key_file = '../Datasets/DFADD/DATASET_GradTTS/test.txt'
    elif args.dataset == 'D2':
        cm_key_file = '../Datasets/DFADD/DATASET_NaturalSpeech2/test.txt'
    elif args.dataset == 'D3':
        cm_key_file = '../Datasets/DFADD/DATASET_StyleTTS2/test.txt'
    elif args.dataset == 'F1':
        cm_key_file = '../Datasets/DFADD/DATASET_MatchaTTS/test.txt'
    elif args.dataset == 'F2':
        cm_key_file = '../Datasets/DFADD/DATASET_PflowTTS/test.txt'

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
    # task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}_{args.fixed_length_samples}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence, args.dataset) # test

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

    summarize_data = []

    if not eval_files:
        print(f"No evaluation result files found in {task_out_fold}, skipping scoring.")
    else:
        print(f"Found {len(eval_files)} evaluation result(s). Starting scoring...")
        for i, score_file in enumerate(eval_files):
            base_name = os.path.splitext(os.path.basename(score_file))[0]
            output_txt = os.path.join(task_out_fold, f"score_{base_name}.txt")
            eer_result = eval_to_score_file(score_file, cm_key_file)
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(str(eer_result))
            print(f"Saved score result → {output_txt}")
            
            summarize_data.append((i, eer_result))
    
    if len(summarize_data) == 5:
        levels = [str(level) for level, _ in summarize_data]
        eer_values = [f"{eer * 100:.2f}" for _, eer in summarize_data]

        summarize_csv = os.path.join(task_out_fold, "summarize.csv")
        
        with open(summarize_csv, 'w', encoding='utf-8', newline='') as f:
            f.write("level," + ",".join(levels) + "\n")
            f.write("EER," + ",".join(eer_values) + "\n")
            
        print(f"Saved summary → {summarize_csv}")
        # print("Summary content:")
        # print("level," + ",".join(levels))
        # print("EER," + ",".join(eer_values))
    else:
        print(f"Warning: Expected 5 results (levels 0-4), but got {len(summarize_data)}. Summary not generated.")
        
if __name__ == '__main__':
    main()