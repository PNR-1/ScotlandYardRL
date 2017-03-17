import time
import q_l
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import numpy as np
import matplotlib.pyplot as plt


def run_agent():
    mdx = mdX.Model()

    mdd= mdD.Model()

    epsilon = np.linspace(1, 0, num = 2000)
    start = 0

    start = 0
    countD = 0
    countX = 0
    diff = [0]*2000
    zeros = [0]*2000
    for i in range(len(epsilon)):
        ag = q_l.q_learn(mdx,mdd,explore = epsilon[start])
        reward,mdx,mdd = ag.run_episode()
        if reward < 0:
            countX+=1
        else:
            countD+=1

        print(epsilon[start],'\t',countD,'\t',countX)
        start = start + 1
        diff[i] = countD-countX
    print("Detectives=",countD)
    print("X=",countX)
    plt.plot(diff)
    plt.plot(zeros,'r--')
    plt.ylabel('Detectives- X')
    plt.xlabel('Episode')
    plt.show()


run_agent()
