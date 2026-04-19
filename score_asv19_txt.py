import numpy as np
import argparse
import sys
import glob
import os
import re
from importlib import import_module
from contextlib import redirect_stdout


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,
      Speech waveform -> [CM] -> [ASV] -> decision
    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.
    INPUTS:
      bonafide_score_cm   A vector of POSITIVE CPASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CPASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.
                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.
      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?
    OUTPUTS:
      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).
    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.
    References:
      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)
      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print(
            't-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(
            cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds


def eerandtdcf(score_file, label_file, asv_label):
    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }
    asv_data = np.genfromtxt(asv_label, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    target = []
    nontarget = []
    target_score = []
    nontarget_score = []
    wav_lists = []
    score = {}
    lable_list = {}
    wrong = 0
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                label = line[4]
                lable_list[wav_id] = label
                if label == "spoof":
                    nontarget.append(wav_id)
                else:
                    target.append(wav_id)

    with open(score_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0]
                wav_lists.append(wav_id)
                score[wav_id] = float(line[3]) # 根据文件内容取出所在列的结果
    for wav_id in target:
        # target_score.append((score[wav_id]))
        target_score.append(float(score[wav_id]))
    for wav_id in nontarget:
        # nontarget_score.append((score[wav_id]))
        nontarget_score.append(float(score[wav_id]))
    target_score = np.array(target_score)
    nontarget_score = np.array(nontarget_score)
    eer_cm, Threshhold = compute_eer(target_score, nontarget_score)
    '''
    print("EER={}, Threshhold={}".format(EER, Threshhold))
    for wav_id in wav_lists:
        if float(score[wav_id])>Threshhold and lable_list[wav_id]=="spoof":
            wrong+=1

    acc=(len(score)-wrong)/len(score)
    print("Acc={}".format(acc))
    '''
    tDCF_curve, CM_thresholds = compute_tDCF(target_score, nontarget_score, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                                             cost_model, False)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    print('ASV SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
    print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
    print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
    print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))
    # 只使用 CM 的结果
    print('\nCM SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

    print('\nTANDEM')
    print('   min-tDCF       = {:8.5f}'.format(min_tDCF))


def main():
    
    parser = argparse.ArgumentParser(description='Get ASVspoof2019 EER and min-tDCF scores')
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument('--block-layers', type=int, required=True, help='Total number of blocks in the model')
    parser.add_argument('--block-sequence', type=str, required=True, help='block sequence for Mamba2 model, ref. class StackedMamba2')
    parser.add_argument('--block-cls', type=str, required=True, default='Mamba2', help='Only Mamba2 / Hydra / GDN block types are supported')
    parser.add_argument('--n-mamba', type=int, default=1, help='Number of mixed SSM component, default is 1')
    parser.add_argument("--out-fold", type=str, default='./outputs/', help="output folder")
    parser.add_argument("--dataset", type=str, help="dataset name", required=True, choices=['asv19'])
    parser.add_argument('--algo', type=int, default=0, required=True, 
                    help='Rawboost algos discriptions. Follow two diff. setting (3 for DF, 5 for LA) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    parser.add_argument('--freeze', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    parser.add_argument('--dfadd-subset', type=str, default=None, help='Support: GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2')

    
    args = parser.parse_args()
    
    labelFile = "../Datasets/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    asvlabel = "../Datasets/ASVspoof2019/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    
    config_module = import_module('train_config') # 由训练配置决定，保持训练、测试、评分路径一致
    task_config = config_module.TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Unknown task: {args.task}. Please define it in train_config.py")
    # task_out_fold = os.path.join(args.out_fold, args.task) # "./outputs/{task name}/"
    b_repeat = args.block_layers
    b_sequence = args.block_sequence
    if "n_" in b_sequence:
        b_sequence = b_sequence.replace("n_", str(args.n_mamba))
    mixer_type = args.block_cls
    fz_lb = "freeze" if args.freeze else "finetune"
    task_out_fold = os.path.join(args.out_fold, args.task, f"repeats_{b_repeat}", f"{mixer_type}_algo{args.algo}_{fz_lb}", b_sequence, args.dataset)

    eval_files = [] # 需要评分的文件列表
    
    # loss 最低模型结果
    best_eval_file = os.path.join(task_out_fold, "eval_model_best.txt")
    if os.path.exists(best_eval_file):
        eval_files.append(best_eval_file)
    else:
        print(f"Warning: Not found: {best_eval_file}")
    
    # 添加所有 epoch 模型结果 / checkpoint
    epoch_pattern = os.path.join(task_out_fold, "eval_best_*.txt")
    epoch_files = glob.glob(epoch_pattern)
    if not epoch_files:
        print(f"Info: No 'eval_best_*.txt' files found in {task_out_fold}")
    else:
        # 按 epoch 数字排序
        def extract_epoch(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r"eval_best_(\d+)\.txt", filename)
            return int(match.group(1)) if match else -1

        epoch_files = sorted(epoch_files, key=extract_epoch)
        eval_files.extend(epoch_files)

    # 按照文件序列逐个评分
    if not eval_files:
        print(f"No evaluation result files found in {task_out_fold}, skipping scoring.")
    else:
        print(f"Found {len(eval_files)} evaluation result(s). Starting scoring...")
        for scoreFile in eval_files:
            base_name = os.path.splitext(os.path.basename(scoreFile))[0]
            output_txt = os.path.join(task_out_fold, f"score_{base_name}.txt")
            with open(output_txt, 'w', encoding='utf-8') as f:
                # 将函数print的输出重定向到文件
                with redirect_stdout(f):
                    try:
                        eerandtdcf(scoreFile, labelFile, asvlabel)
                    except Exception as e:
                        print(f"Error occurred while scoring {base_name}: {e}")
            print(f"Saved score result → {output_txt}")    
    
            
if __name__ == '__main__':
    main()