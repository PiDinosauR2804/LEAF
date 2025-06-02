source activate zhangchenlong

for i in ACE MAVEN
do
    for j in 42
    do
        for k in 5 10
        do
            if [ "$i" = "ACE" ]; then
                t=10
            else
                t=20
            fi

            python main.py \
                --data_root ./data/data_ids_enhence \
                --dataset $i \
                --perm_id 0 \
                --seed $j \
                --shot_num $k \
                --class_num $t \
                --backbone bert-base-uncased \
                --lr 2e-5 \
                --decay 1e-4 \
                --no_freeze_bert \
                --batch_size 4 \
                --device cuda:0 \
                --log \
                --log_dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                --log_name ashuffle_lnone_r1 \
                --wandb \
                --project_name ACE_test_skip \
                --save_output output_logg \
                --dweight_loss \
                --rep_aug mean \
                --distill mul \
                --single_label \
                --cl_aug shuffle \
                --aug_repeat_times 10 \
                --joint_da_loss ce \
                --sub_max \
                --cl_temp 0.07 \
                --ucl \
                --skip_first_cl ucl+tlcl \
                --use_description \
                --num_description 3 \
                --ratio_loss_des_cl 0.1 \
                --epochs 3 \
                --task_ep_time 6 \
                --uniform_ep 1 \
                --eval_freq 2 \
                --skip_eval_ep 0 \
                --patience 4 \
                --lora_rank 64 \
                --lora_alpha 64 \
                --lora_dropout 0.3 \
                --use_lora \
                --use_mole \
                --mole_num_experts 4 \
                --mole_top_k 2 \
                --use_general_expert \
                --entropy_weight 0.1 \
                --load_balance_weight 1 \
                --general_expert_weight 0.2 \
                --step_size 1 \
                --gammalr 0.99 \
                --eval_batch_size 256 \
                --eval_ratio 0.25 \
                --gpt_augmention \
                --decrease_0_gpt_augmention \
                --ratio_loss_gpt 0.1 \
                --use_weight_ce \
                --alpha_ce 0.3 \
                --skip_des
        done
    done
done
