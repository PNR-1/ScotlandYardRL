import time
import q_l
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import numpy as np
mdx = mdX.Model()
mdd = [None] * 5
for i in range (5):
	mdd[i] = mdD.Model()

epsilon = np.linspace(1, 0, num = 5000)
start = 0

def run_agent():
    start = 0
    for i in range(len(epsilon)):
        times = time.time()
        _,month,day,hour,minute,second,_,_,_ = time.localtime(time.time())
        directory = str(month) + '-' + str(day) + '/' + str(hour) + '/' + str(minute) + '/'
        file_name = str(second)+'-'+ str(times - int(times)) + '.txt'
        ag = q_l.q_learn(mdx,mdd,explore = epsilon[start],directory = directory,file_name=file_name,loglevel='DEBUG')
        print(epsilon[start],'\t',ag.run_episode())
        start = start + 1

run_agent()
