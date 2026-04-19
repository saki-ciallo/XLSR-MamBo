# train_test_score_AIO.py

import argparse
import subprocess
import sys


def run_tasks(
        epochs: int,
        types: str, 
        dataset: str, 
        task: str, 
        fixed_length_samples: int,
        block_cls: str, 
        block_layers: int, 
        block_sequence: str, 
        n_mamba: int, 
        save_checkpoint: bool, 
        algo: int, 
        freeze: bool, 
        save_model: bool,
        dfadd_subset: str, 
        count_params_only: bool,
        gpu_id: int, 
        out_fold: str, 
    ):
    scripts = {
        'train': 'train.py',
        'test':  'eval_asv_txt.py', # 兼容asv19/asv21la/asv21df/itw
        'score_asv19': 'score_asv19_txt.py',
        'score_asv21': 'score_asv21_txt.py',
        'score_itw': 'score_itw_txt.py',
        'score_dfadd': 'score_dfadd_txt.py',
    }
    
    if types == "score":
        if dataset == "asv19":
            score_script = scripts['score_asv19']
        elif dataset in ["asv21la", "asv21df"]:
            score_script = scripts['score_asv21']
        elif dataset == "itw":
            score_script = scripts['score_itw']
        elif dataset in ["D1", "D2", "D3", "F1", "F2"]:
            score_script = scripts['score_dfadd']
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        script = score_script
    else:
        script = scripts[types] # train 或 test 脚本
        
    extra = ["--save-checkpoint"] if save_checkpoint else [] # only for train
    extra_count_params = ["--count-params-only"] if count_params_only else []
    extra_freeze = ["--freeze"] if freeze else []
    extra_save_model = ["--save-model"] if save_model else []
    
    print(f"[{types.upper()}] {task}")
    if types == "train":
        cmd = ["python", script, "--task", task, "--block-layers", str(block_layers), "--block-sequence", block_sequence, "--block-cls", block_cls, "--n-mamba", str(n_mamba), "--out-fold", out_fold, "--dataset", dataset, "--gpu-id", str(gpu_id), "--algo", str(algo), "--epochs", str(epochs), "--dfadd-subset", str(dfadd_subset), "--fixed-length-samples", str(fixed_length_samples)] + extra + extra_count_params + extra_freeze + extra_save_model
    elif types == "test":
        cmd = ["python", script, "--task", task, "--block-layers", str(block_layers), "--block-sequence", block_sequence, "--block-cls", block_cls, "--n-mamba", str(n_mamba), "--out-fold", out_fold, "--dataset", dataset, "--gpu-id", str(gpu_id), "--algo", str(algo), "--epochs", str(epochs), "--dfadd-subset", str(dfadd_subset), "--fixed-length-samples", str(fixed_length_samples)] + extra_freeze
    elif types == "score":
        cmd = ["python", script, "--task", task, "--block-layers", str(block_layers), "--block-sequence", block_sequence, "--block-cls", block_cls, "--n-mamba", str(n_mamba), "--out-fold", out_fold, "--dataset", dataset, "--algo", str(algo), "--dfadd-subset", str(dfadd_subset), "--fixed-length-samples", str(fixed_length_samples)] + extra_freeze
        
    try:
        result = subprocess.run(cmd) # 要求参数使用 str 类型传入，不影响脚本需要的传入类型
    except Exception as e:
        print(f"  [Exception] An error occurred while running the task: {e}")
        return False
    if result.returncode == 0:
        print(f"{task} {types}ed successfully.")
        
    return True

def main():
    # 通用设定
    parser = argparse.ArgumentParser(description="Train, Test and Score model for ASV tasks")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument("--type", type=str, choices=['train','test','score'], required=True, help="Operation types: train, test, or score")
    parser.add_argument("--dataset", type=str, required=True, default='asv19', help="Datasets name: asv19, asv21la, asv21df, itw, D1, D2, D3, F1, F2. The asv19 for training") 
    parser.add_argument("--task", type=str, required=True, help="The SSM four types: xlsr_Mamba1, xlsr_Mamba2, xlsr_Hydra, xlsr_GDN")
    parser.add_argument('--fixed-length-samples', type=int, default=66800, help='Fixed length of audio samples')
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba1', help='The SSM types: Mamba1, Mamba2, Hydra, GDN')
    parser.add_argument('--block-layers', type=int, required=True, help='The number of block layers')
    parser.add_argument('--block-sequence', type=str, required=True, help='The block sequence')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    parser.add_argument('--save-checkpoint', action='store_true', help='save periodic checkpoints')
    parser.add_argument('--algo', type=int, default=0, required=True, 
                    help='Rawboost algos discriptions. Setting 3 for DF, 5 for LA. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--freeze', action="store_true", help='Whether to fine-tune the XLS-R or not')
    parser.add_argument('--save-model', action='store_true', help='save best model')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Use DFADD subset traning, support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')
    parser.add_argument("--count-params-only", action="store_true", help="Count model parameters and exit")
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use, allowing each script to specify different GPUs')
    parser.add_argument("--out-fold", type=str, default='./outputs/', help="output folder")

    args = parser.parse_args()

    task = args.task
    print(f"Tasks: {task}")
        
    success = run_tasks(
                    epochs=args.epochs,
                    types=args.type, 
                    dataset=args.dataset, 
                    task=task, 
                    fixed_length_samples=args.fixed_length_samples,
                    block_cls=args.block_cls, 
                    block_layers=args.block_layers, 
                    block_sequence=args.block_sequence, 
                    n_mamba=args.n_mamba, 
                    save_checkpoint=args.save_checkpoint, 
                    algo=args.algo, 
                    freeze=args.freeze, 
                    save_model=args.save_model,
                    dfadd_subset=args.dfadd_subset, 
                    count_params_only=args.count_params_only,
                    gpu_id=args.gpu_id, 
                    out_fold=args.out_fold,
    )  
    sys.exit(0 if success else 1)
    
    
if __name__ == '__main__':
    main()
