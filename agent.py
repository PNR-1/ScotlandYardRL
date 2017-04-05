import time
import q_l
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import numpy as np


def run_agent():
    mdx = mdX.Model()
    # mdd = mdD.Model()
    mdd = [None] * 5
    for i in range (5):
        mdd[i] = mdD.Model()

    epsilon = np.linspace(1, 0, num = 25001)
    start = 0
    countD = 0
    countX = 0

    print("Training")
    print("Run\tD_wins\tX_wins\n")
    for i in range(len(epsilon)):
        # times = time.time()
        # _,month,day,hour,minute,second,_,_,_ = time.localtime(time.time())
        # directory = str(month) + '-' + str(day) + '/' + str(hour) + '/' + str(minute) + '/'
        # file_name = str(second)+'-'+ str(times - int(times)) + '.txt'
        ag = q_l.q_learn(mdx,mdd,explore = epsilon[start])
        reward, mdx, mdd = ag.run_episode()
        #ag.close_log()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(start%1==0):
            #wins.write(str(start)+"\t"+str(countD)+"\t"+str(countX)+"\n")
        print(str(epsilon[start])+','+str(countD)+','+str(countX))
        start = start + 1
        #diff[i] = countD-countX
    print("X=",countX)
    print("Detectives=", countD)
    #wins.write("Detectives="+str(countD)+"\n")
    #wins.write("X="+str(countX))
    #ch.close
    #wins.close
    start = 0
    countD = 0
    countX = 0
    #wins = open("test.txt", "w+")
    print("-------------------------------------")
    print("Testing")
    print("Run\tD_wins\tX_wins\n")
    for i in range(len(epsilon)):
        #times = time.time()
        #_,month,day,hour,minute,second,_,_,_ = time.localtime(time.time())
        #directory = str(month) + '-' + str(day) + '/' + str(hour) + '/' + str(minute) + '/'
        #file_name = str(second)+'-'+ str(times - int(times)) + '.txt'
        ag = q_l.q_learn(mdx,mdd,explore = 0)
        reward, mdx, mdd= ag.run_episode()
        #ag.close_log()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(start%500==0):
        print(str(epsilon[start])+','+str(countD)+','+str(countX))
        start = start + 1

    print("X=",countX)
    print("Detectives=", countD)
    #wins.write("Detectives=",countD)
    #wins.write("X=",countX)
    #wins.close





run_agent()
