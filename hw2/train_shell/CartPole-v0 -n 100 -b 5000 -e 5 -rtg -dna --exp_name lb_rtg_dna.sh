cd ~/PycharmProjects/homework/hw2
CUDA_VISIBLE_DEVICES=0 python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna