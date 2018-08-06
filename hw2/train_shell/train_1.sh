cd ~/PycharmProjects/homework/hw2
CUDA_VISIBLE_DEVICES=3 python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna