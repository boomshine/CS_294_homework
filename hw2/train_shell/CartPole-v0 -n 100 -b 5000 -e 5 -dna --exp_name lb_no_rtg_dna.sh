cd ~/PycharmProjects/homework/hw2
CUDA_VISIBLE_DEVICES=1 python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna
