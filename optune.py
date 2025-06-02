from classifier.train import train
from configs import parse_arguments
import optuna
import wandb
import torch
from loguru import logger

args = parse_arguments()

def objective(trial):
    # Hằng 
    args.epochs = 50
    args.use_lora = True
    args.logs_dir = "logs/classifier"
    args.data_root = "./data/data_ids"
    args.dataset = "MAVEN"
    args.backbone = "bert-base-uncased"
    args.decay = 1e-4
    args.no_freeze_bert = True
    args.shot_num = 5
    args.device = "cuda:0"
    args.log = True
    args.log_dir = "./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/"
    args.log_name = "ashuffle_lnone_r1"
    args.dweight_loss = True
    args.rep_aug = "mean"
    args.distill = "pd"
    args.class_num = 20
    args.single_label = True
    args.cl_aug = "shuffle"
    args.aug_repeat_times = 3
    args.joint_da_loss = "ce"
    args.sub_max = True
    args.cl_temp = 0.07
    args.tlcl = False
    args.ucl = True
    args.skip_first_cl = "ucl+tlcl"
    args.use_description = True
    args.num_description = 3
    args.ratio_loss_des_cl = 1
    args.task_ep_time = 1
    args.early_stop = True
    args.skip_eval_ep = 0
    args.eval_freq = 1
    args.patience = 4
    args.early_stop = True
    args.classifier_layer = 1
    args.hidden_dim = 128
    args.dropout = 0.5
    args.use_general_expert = True
    args.use_mole = True
    args.step_size = 1
    args.wandb = True
    args.eval_batch_size = 64
    args.task_ep_time = 1
    args.uniform_ep = 3
    args.lora_dropout = 0.3
    args.batch_size = 8
    args.project_name = "HANet_mole_bert_full"

    # Tham số cho Optuna trial
    args.lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
    args.lora_rank = trial.suggest_categorical("lora_rank", [64, 128, 256])
    args.lora_alpha = trial.suggest_int("lora_alpha", 32, 128, step=32)
    args.mole_num_experts = trial.suggest_categorical("mole_num_experts", [4, 8])
    args.mole_top_k = trial.suggest_categorical("mole_top_k", [2, 4])
        
    args.gammalr = trial.suggest_float("gamma", 0.8, 1.0, step=0.01)
    args.entropy_weight = trial.suggest_float("entropy_weight", 0.1, 1, step=0.1)
    args.load_balance_weight = trial.suggest_float("load_balance_weight", 0.1, 1.0, step=0.1)
    args.general_expert_weight = trial.suggest_float("general_expert_weight", 0.1, 1.0, step=0.1)
    
    try:
        f1 = train(0, args, trial)
        return f1
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA out of memory. Releasing GPU memory...")
            torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
            return float('inf')  # Giá trị lỗi để Optuna bỏ qua
        else:
            raise e  # Nếu là lỗi khác, vẫn cho nó raise lên


if __name__ == "__main__":
    wandb.login()
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))