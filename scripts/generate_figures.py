#!/usr/bin/env python3
from pathlib import Path
import json,sys,torch
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rlqls import build_cah16_surrogate,build_h3o130_surrogate,RLQLSEnv
from rlqls.dqn import QNetwork
from rlqls.evaluation import evaluate_network_batched,evaluate_sweeping_batched
from rlqls.plotting import moving_average
OUT=ROOT/'results'/'figures';OUT.mkdir(parents=True,exist_ok=True)

def load_network(path,model):
    p=torch.load(path,map_location='cpu',weights_only=False)
    n=QNetwork(model.n_states,model.n_actions,tuple(p['config']['hidden_sizes']));n.load_state_dict(p['online']);n.eval();return n,p

# CaH+
cah=build_cah16_surrogate();env=RLQLSEnv(cah,max_steps=100)
net,p=load_network(ROOT/'results/checkpoints/cah_qmdp_1000_seed7.pt',cah)
rl=evaluate_network_batched(env,net,episodes=3000,seed=10000);sw=evaluate_sweeping_batched(env,episodes=3000,seed=20000)
x=np.arange(1,31)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[rl.completion_fraction(int(v)) for v in x],label='reconstruction: qMDP DQN')
ax.plot(x,[sw.completion_fraction(int(v)) for v in x],label='reconstruction: sweeping')
paper_x=np.array([2,3,4,5,6,7,8,18]);paper_rl=np.array([0,.15,.35,.35,.35,.45,.56,.99]);paper_sw=np.array([0,0,.09,.34,.47,.47,.47,.94])
ax.scatter(paper_x,paper_rl,marker='o',label='paper: RL');ax.scatter(paper_x,paper_sw,marker='x',label='paper: sweeping')
ax.set(xlabel='pulse-measurement steps',ylabel='fraction completed',ylim=(0,1.02),title='CaH+ J={1,2}: completion probability');ax.legend();fig.tight_layout();fig.savefig(OUT/'cah_completion.png',dpi=200);plt.close(fig)
lengths=p['history']['episode_lengths'];mx,my=moving_average(lengths,100)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(lengths)+1),lengths,alpha=.2,label='episode');ax.plot(mx,my,label='100-episode mean')
ax.set(xlabel='training episode',ylabel='steps',title='CaH+ qMDP training');ax.legend();fig.tight_layout();fig.savefig(OUT/'cah_training.png',dpi=200);plt.close(fig)

# H3O+ (paper-faithful 4-motional-state reconstruction, short run)
h3o=build_h3o130_surrogate(ROOT/'data');envh=RLQLSEnv(h3o,max_steps=150,overlap_penalty=1.0)
neth,ph=load_network(ROOT/'results/checkpoints/h3o_4mot_qmdp_100_seed41.pt',h3o)
rlh=evaluate_network_batched(envh,neth,episodes=200,seed=130000);swh=evaluate_sweeping_batched(envh,episodes=200,seed=140000)
x=np.arange(1,151)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[rlh.completion_fraction(int(v)) for v in x],label='reconstruction: qMDP DQN (100 episodes)')
ax.plot(x,[swh.completion_fraction(int(v)) for v in x],label='reconstruction: sweeping')
ax.scatter([62,150],[.80,.934],marker='o',label='paper reported RL anchors')
ax.set(xlabel='pulse-measurement steps',ylabel='fraction completed',ylim=(0,1.02),title='H3O+: partial reconstruction');ax.legend();fig.tight_layout();fig.savefig(OUT/'h3o_completion.png',dpi=200);plt.close(fig)
lengths=ph['history']['episode_lengths'];mx,my=moving_average(lengths,20)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(lengths)+1),lengths,alpha=.25,label='episode');ax.plot(mx,my,label='20-episode mean')
ax.set(xlabel='training episode',ylabel='steps',title='H3O+ qMDP training (partial)');ax.legend();fig.tight_layout();fig.savefig(OUT/'h3o_training.png',dpi=200);plt.close(fig)

print(OUT)
