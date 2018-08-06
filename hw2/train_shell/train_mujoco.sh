cd ~/PycharmProjects/homework/hw2
CUDA_VISIBLE_DEVICES=3 python train_pg.py CartPole-v0 -n 100 -bl -b 5000 -e 5 -rtg -dna --exp_name GAE_rtg_dna