import time
import os
import gc
import math
import random
import argparse
import sys
import torch
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout
from importlib import import_module
from addition_loss.focal_loss import FocalLoss
from dataset_load import asv19_dataloader, dfadd_dataloader

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print("                  Model Parameters Count")
    print("="*60)
    print(f"Total params      : {total_params:,}")
    print(f"   ≈ {total_params / 1e6:.3f} M parameters in total")
    print(f"Trainable     : {trainable_params:,}")
    print(f"   ≈ {trainable_params / 1e6:.3f} M parameters to train")
    print(f"Frozen      : {(total_params - trainable_params):,}")
    print("="*60)
            

def get_cosine_schedule_with_warmup_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Linear Warmup + Cosine Decay
    
    Parameters:
    - optimizer: 传入优化器
    - num_warmup_steps: warmup 步数
    - num_training_steps: 总训练步数
    - min_lr_ratio: 最终学习率与初始学习率的比值
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def train(args, model, device, train_loader, optimizer, criterion, epoch, scheduler=None, use_cosine_warmup=False):
    model.train()
    
    train_loss = 0.0
    num_total = 0.0
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    
    num_batch = len(train_loader) 
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', 
                       total=num_batch,leave=False)
    
    for batch_x, batch_y in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # batch_out = model(batch_x)
        # batch_loss = criterion(batch_out, batch_y)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
        
        optimizer.zero_grad(set_to_none=True) # 如有兼容问题使用默认
        # optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # Cosine Warmup: 每个 step 更新学习率
        if use_cosine_warmup and scheduler is not None:
            scheduler.step()

        # 累计统计
        train_loss += (batch_loss.item() * batch_size)
        
        # 在进度条中显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

    train_loss /= num_total
    # sys.stdout.flush()
    return train_loss

def valid(model, device, valid_loader, criterion, epoch):
    model.eval()
    
    val_loss = 0
    num_total = 0.0
    correct = 0
    
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    progress_bar = tqdm(valid_loader, desc='Validation Testing',leave=False)
    with torch.no_grad():
        for batch_x, batch_y in progress_bar:
            batch_size = batch_x.size(0)
            target = torch.LongTensor(batch_y).to(device)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # batch_out = model(batch_x)
            # batch_loss = criterion(batch_out, batch_y)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)

            pred = batch_out.max(1)[1] 
            correct += pred.eq(target).sum().item()

            val_loss += (batch_loss.item() * batch_size)
            
    val_loss /= num_total  # 平均每个样本的loss
    test_accuracy = 100. * correct / len(valid_loader.dataset)

    print(f'Epoch {epoch}: Valid set: Average loss: {val_loss:.6f}, Accuracy: {correct}/{len(valid_loader.dataset)} ({test_accuracy:.2f}%)\n')

    return val_loss, test_accuracy


def reproducibility(random_seed):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to True")
    print("cudnn_benchmark set to False")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return



def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        current = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current)
        print(f"Force set to GPU {current}: {name}")
        return torch.device(f'cuda:{current}')
    else:
        raise EnvironmentError("No GPU available, please run on a machine with CUDA-enabled GPU.")

def save_model(model, path):
    if hasattr(model, '_orig_mod'):
        torch.save(model._orig_mod.state_dict(), path) # 如果使用编译，保存原始模型
    else:
        torch.save(model.state_dict(), path)

def main():
    # 通用设定
    parser = argparse.ArgumentParser(description='Universal function approximator training for various tasks')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument("--dataset", type=str, required=True, default='asv19', help="Datasets name: asv19, asv21la, asv21df, itw, D1, D2, D3, F1, F2. The asv19 for training")
    parser.add_argument("--task", type=str, required=True, help="The SSM four types: xlsr_Mamba1, xlsr_Mamba2, xlsr_Hydra, xlsr_GDN")
    parser.add_argument('--fixed-length-samples', type=int, default=66800, help='Fixed length of audio samples')
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba1', help='The SSM types: Mamba1, Mamba2, Hydra, GDN')
    parser.add_argument('--block-layers', type=int, required=True, help='The number of block layers')
    parser.add_argument('--block-sequence', type=str, required=True, help='The block sequence')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    parser.add_argument('--save-checkpoint', action='store_true', help='save periodic checkpoints')
    parser.add_argument("--count-params-only", action="store_true", help="Count model parameters and exit")
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--save-model', action='store_true', help='save best model')
    parser.add_argument('--freeze', action="store_true", help='Whether to fine-tune the XLS-R or not')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Use DFADD subset traning, support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument("--out-fold", type=str, default='./outputs/', help="output folder", )
    
    
    # 参考 https://github.com/TakHemlata/RawBoost-antispoofing/blob/main/main.py 使用默认值
    ##===================================================Rawboost data augmentation parameters======================================================================#

    parser.add_argument('--algo', type=int, default=0, required=True, 
                    help='Rawboost algos discriptions. Setting 3 for DF, 5 for LA. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    ##===================================================Rawboost data augmentation ======================================================================#
    
    args = parser.parse_args()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device(gpu_id=args.gpu_id) # not consider cpu and multi-gpu for now
    print(f"Using device: {device}")
    
    torch.set_float32_matmul_precision('high')
        
    reproducibility(args.seed)
    
    
    # 导入任务配置
    config_module = import_module('train_config')  # 导入 train_config.py
    task_config = config_module.TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Unknown task: {args.task}. Please define it in train_config.py")

    # 创建目录
    b_repeat = args.block_layers
    b_sequence = args.block_sequence
    if "n_" in b_sequence:
        b_sequence = b_sequence.replace("n_", str(args.n_mamba))
    mixer_type = args.block_cls
    fz_lb = "freeze" if args.freeze else "finetune"
    dfadd_subset = "" if args.dfadd_subset == "None" else f"_{args.dfadd_subset}"
    task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}{dfadd_subset}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence) # 任务目录
    
    # task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}_{args.fixed_length_samples}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence) # only for test the code
    
    os.makedirs(task_out_fold, exist_ok=True)
    
    # 模型
    model = task_config.model_classifier(task_config.model_config,
                                        freeze=args.freeze,
                                        block_layers=args.block_layers,
                                        block_sequence=args.block_sequence,
                                        n_mamba=args.n_mamba,
                                        block_cls=args.block_cls,
                                        fixed_length_samples=args.fixed_length_samples
                                        ).to(device) # mb中处理mamba2_config
    if args.count_params_only:
        params_txt_path = os.path.join(task_out_fold, "model_parameters.txt")
        print(f"Compute and save the parameters count to a text file: {params_txt_path}")
        with open(params_txt_path, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                count_parameters(model)
        sys.exit(0)
        
    # model = torch.compile(model)
    model = torch.compile(model, mode="max-autotune-no-cudagraphs") # 效率略有提升
    
    # 对损失日志文件初始化，写入表头
    loss_log = os.path.join(task_out_fold, 'train_valid_log.csv')
    if not os.path.exists(loss_log):
        with open(loss_log, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,valid_loss,valid_accuracy,learning_rate\n')
    
    if args.save_checkpoint:
        os.makedirs(os.path.join(task_out_fold, 'checkpoint'), exist_ok=True)
        print(f"Use checkpoint")

    if args.dfadd_subset != "None":
        print(f"Use DFADD training set to train: {args.dfadd_subset}")
        train_loader = dfadd_dataloader(types="train", subset=args.dfadd_subset, batch_size=32, num_workers=10, args_main=args)
        dev_loader = dfadd_dataloader(types="valid", subset=args.dfadd_subset, batch_size=32, num_workers=10, args_main=args)
    else: 
        print(f"use ASVspoof2019 dataset to train")
        train_loader = asv19_dataloader(types="train", batch_size=32, num_workers=10, args_main=args)
        dev_loader = asv19_dataloader(types="dev", batch_size=32, num_workers=10, args_main=args)

    print(f'Train: {len(train_loader.dataset)} samples')
    print(f'Dev:   {len(dev_loader.dataset)} samples')
    
     
    optimizer = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, model.parameters()),
        model.parameters(),
        lr=1e-5, betas=(0.9, 0.95),
        weight_decay=0.05
        )
    
    criterion = FocalLoss(gamma=2, alpha=[0.1, 0.9], task_type='multi-class', num_classes=2)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    
    use_cosine_warmup = True
    if use_cosine_warmup:
        # Linear Warmup 10% + Cosine Decay to 10%
        scheduler = get_cosine_schedule_with_warmup_min_lr(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.1,
        )
        print(f"Using Cosine Schedule: {warmup_steps} warmup steps ({warmup_steps/total_steps*100:.1f}%), "
              f"decay to 10% of base lr")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-8
        )

    patience = 7
    not_improved = 0
    best_loss = float('inf')
    bests = []
    
    print(f'\nStart training loop, total epoch: {args.epochs}\n')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, criterion, epoch, 
                          scheduler=scheduler, use_cosine_warmup=use_cosine_warmup)
        valid_loss, valid_accuracy = valid(model, device, dev_loader, criterion, epoch)
    
        # ReduceLROnPlateau: 每个 epoch 根据验证集 loss 调整
        if not use_cosine_warmup:
            scheduler.step(valid_loss)

        # write loss to log file
        current_lr = optimizer.param_groups[0]['lr']
        with open(loss_log, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss},{valid_loss},{valid_accuracy},{current_lr}\n')

        # update best model
        if args.save_model and valid_loss < best_loss:
            best_loss = valid_loss
            save_model(model, os.path.join(task_out_fold, 'model_best.pt'))
            # torch.save(model.state_dict(), os.path.join(task_out_fold, 'model_best.pt'))
            torch.save(optimizer.state_dict(), os.path.join(task_out_fold, 'optimizer_best.pt'))
            print("New GLOBAL BEST", end=' ')
            not_improved = 0
        else:
            not_improved += 1
        
        # save checkpoint if it's in the top-5 bests
        if args.save_checkpoint:
            n_best = 5
            # find the position to insert current valid_loss into bests
            insert_pos = None
            for i in range(n_best):
                if i >= len(bests) or valid_loss < bests[i][0]:
                    insert_pos = i
                    break
            
            if insert_pos is not None:
                # 只有当插入位置不是最后一位时才需要重命名 
                if insert_pos < n_best - 1:
                    # 将后面的模型文件依次重命名(从后往前)
                    # 注意: 只重命名到第n_best-2位,因为第n_best-1位会被淘汰或覆盖
                    for t in range(min(len(bests), n_best - 1) - 1, insert_pos - 1, -1):
                        # 重命名模型文件
                        old_model = os.path.join(task_out_fold, 'checkpoint', f'best_{t}.pt')
                        new_model = os.path.join(task_out_fold, 'checkpoint', f'best_{t+1}.pt')
                        if os.path.exists(old_model):
                            # os.remove(new_model) 否则需要先删除旧文件再rename
                            os.rename(old_model, new_model) 
                        # 重命名优化器文件
                        old_opt = os.path.join(task_out_fold, 'checkpoint', f'opt_{t}.pt')
                        new_opt = os.path.join(task_out_fold, 'checkpoint', f'opt_{t+1}.pt')
                        if os.path.exists(old_opt):
                            os.rename(old_opt, new_opt)
                
                # 在insert_pos位置插入新的(loss, epoch)
                bests.insert(insert_pos, (valid_loss, epoch))
                bests = bests[:n_best]  # 保持列表最多5个元素
                
                # 保存当前模型到位置insert_pos (如果是第5位则直接覆盖)
                # torch.save(model.state_dict(), os.path.join(task_out_fold, 'checkpoint', f'best_{insert_pos}.pt'))
                save_model(model, os.path.join(task_out_fold, 'checkpoint', f'best_{insert_pos}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(task_out_fold, 'checkpoint', f'opt_{insert_pos}.pt'))
                print(f"New Top-{insert_pos+1} (epoch {epoch})", end='')

        # early stopping
        print(f"NO IMPROVE: {not_improved}/{patience}")
        if not_improved >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            print(f"Global best valid loss: {best_loss:.6f}")
            break

    if args.save_checkpoint and any(ep > 0 for _, ep in bests):
        top5_txt_path = os.path.join(task_out_fold, 'top5_checkpoints.txt')
        
        with open(top5_txt_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("\n=== Final Top-5 Checkpoints (Lowest Valid Loss) ===")
                for rank, (loss, ep) in enumerate(bests):
                    marker = " (GLOBAL BEST)" if ep > 0 and loss == best_loss else ""
                    print(f"  {rank+1}. best_{rank}.pt  →  epoch {ep:3d}  |  valid loss {loss:.8f}{marker}")
        with open(top5_txt_path, 'r', encoding='utf-8') as f:
            final_summary = f.read()
        print("\n" + final_summary)
        print(f"Top-5 checkpoint has been saved to the path: {top5_txt_path}")


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    s = time.time()
    main()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total-3600*h) // 60
    s = total - 3600*h - 60*m
    print(f"Total training time: {h:02d}:{m:02d}:{s:02d}")
    # cleanup() 没啥用
    