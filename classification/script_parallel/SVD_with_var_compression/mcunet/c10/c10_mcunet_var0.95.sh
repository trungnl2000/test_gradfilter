pwd
date

general_config_args="--config configs_new/mcunet_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/mcunet/cifar10/SVD_dim0/var0.95"
data_args="--data.name cifar10 --data.data_dir data/cifar10 --data.train_workers 24 --data.val_workers 24" #--data.batch_size 1"
trainer_args="--trainer.max_epochs 50"
model_args="--model.SVD_var 0.95 --model.with_SVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 10 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name SVD_l1_var0.95_${usr_group_kl} --model.num_of_finetune 1 
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l2_var0.95_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l3_var0.95_${usr_group_kl} --model.num_of_finetune 3
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l4_var0.95_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l5_var0.95_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l6_var0.95_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l7_var0.95_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l8_var0.95_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l9_var0.95_${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name SVD_l10_var0.95_${usr_group_kl} --model.num_of_finetune 10
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l11_var0.95_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l12_var0.95_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l13_var0.95_${usr_group_kl} --model.num_of_finetune 13
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l14_var0.95_${usr_group_kl} --model.num_of_finetune 14
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l15_var0.95_${usr_group_kl} --model.num_of_finetune 15
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l16_var0.95_${usr_group_kl} --model.num_of_finetune 16
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l17_var0.95_${usr_group_kl} --model.num_of_finetune 17
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l18_var0.95_${usr_group_kl} --model.num_of_finetune 18
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l19_var0.95_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l20_var0.95_${usr_group_kl} --model.num_of_finetune 20
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l21_var0.95_${usr_group_kl} --model.num_of_finetune 21
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l22_var0.95_${usr_group_kl} --model.num_of_finetune 22
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l23_var0.95_${usr_group_kl} --model.num_of_finetune 23
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l24_var0.95_${usr_group_kl} --model.num_of_finetune 24
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l25_var0.95_${usr_group_kl} --model.num_of_finetune 25
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l26_var0.95_${usr_group_kl} --model.num_of_finetune 26
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l27_var0.95_${usr_group_kl} --model.num_of_finetune 27
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l28_var0.95_${usr_group_kl} --model.num_of_finetune 28
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l29_var0.95_${usr_group_kl} --model.num_of_finetune 29
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l30_var0.95_${usr_group_kl} --model.num_of_finetune 30
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l31_var0.95_${usr_group_kl} --model.num_of_finetune 31
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l32_var0.95_${usr_group_kl} --model.num_of_finetune 32
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l33_var0.95_${usr_group_kl} --model.num_of_finetune 33
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l34_var0.95_${usr_group_kl} --model.num_of_finetune 34
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l35_var0.95_${usr_group_kl} --model.num_of_finetune 35
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l36_var0.95_${usr_group_kl} --model.num_of_finetune 36
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l37_var0.95_${usr_group_kl} --model.num_of_finetune 37
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l38_var0.95_${usr_group_kl} --model.num_of_finetune 38
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l39_var0.95_${usr_group_kl} --model.num_of_finetune 39
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l40_var0.95_${usr_group_kl} --model.num_of_finetune 40
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l41_var0.95_${usr_group_kl} --model.num_of_finetune 41
# python trainer_cls.py ${common_args} --logger.exp_name SVD_l42_var0.95_${usr_group_kl} --model.num_of_finetune 42