# coding=utf-8
# 运行环境：Ubuntu16.04 Python3.6.3

import sys
import os
import codecs
import json
#reload(sys)
para = sys.argv[1]

A_dic = {}
B_dic = {}
Count_dic = {}
Pi_dic = {}
word_set = set()
state_list = ['B','M','E','S']
line_num = -1

INPUT_DATA = "CTBtrainingset.txt"
PROB_START = "trainHMM\prob_start.py"   #初始状态概率
PROB_EMIT = "trainHMM\prob_emit.py"     #发射概率
PROB_TRANS = "trainHMM\prob_trans.py"   #转移概率


### train 的函数
def init():  #初始化字典
    #global state_M
    #global word_N
    for state in state_list:
        A_dic[state] = {}
        for state1 in state_list:
            A_dic[state][state1] = 0.0
    for state in state_list:
        Pi_dic[state] = 0.0
        B_dic[state] = {}
        Count_dic[state] = 0

def getList(input_str):  #输入词语，输出状态
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append('S')
    elif len(input_str) == 2:
        outpout_str = ['B','E']
    else:
        M_num = len(input_str) -2
        M_list = ['M'] * M_num
        outpout_str.append('B')
        outpout_str.extend(M_list)  #把M_list中的'M'分别添加进去
        outpout_str.append('E')
    return outpout_str

def Output():   #输出模型的三个参数：初始概率+转移概率+发射概率
    with open(PROB_START,"w") as start_fp:
        with open(PROB_EMIT,"w") as emit_fp:
            with open(PROB_TRANS,"w") as trans_fp:
                print ("len(word_set) = %s " % (len(word_set)))

                for key in Pi_dic:           #状态的初始概率
                    Pi_dic[key] = Pi_dic[key] * 1.0 / line_num
                start_fp.write(json.dumps(Pi_dic))

                for key in A_dic:            #状态转移概率
                    for key1 in A_dic[key]:
                        A_dic[key][key1] = A_dic[key][key1] / Count_dic[key]
                trans_fp.write(json.dumps(A_dic))

                for key in B_dic:            #发射概率(状态->词语的条件概率)
                    for word in B_dic[key]:
                        B_dic[key][word] = B_dic[key][word] / Count_dic[key]
                emit_fp.write(json.dumps(B_dic))



def train():
    with open(INPUT_DATA) as ifp:
        init()
        global word_set   #初始是set()
        global line_num   #初始是-1
    
        for line in ifp:
            line_num += 1
            if line_num % 10000 == 0:
                print (line_num)

            line = line.strip()
            if not line:continue


            word_list = []
            for i in range(len(line)):
                if line[i] == " ":continue
                word_list.append(line[i])
            word_set = word_set | set(word_list)   #训练预料库中所有字的集合


            lineArr = line.split(" ")
            line_state = []
            for item in lineArr:
                line_state.extend(getList(item))   #一句话对应一行连续的状态
            if len(word_list) != len(line_state):
                print >> sys.stderr,"[line_num = %d][line = %s]" % (line_num, line.endoce("utf-8",'ignore'))
            else:
                for i in range(len(line_state)):
                    if i == 0:
                        Pi_dic[line_state[0]] += 1      #Pi_dic记录句子第一个字的状态，用于计算初始状态概率
                        Count_dic[line_state[0]] += 1   #记录每一个状态的出现次数
                    else:
                        A_dic[line_state[i-1]][line_state[i]] += 1    #用于计算转移概率
                        Count_dic[line_state[i]] += 1
                        if not ( word_list[i] in B_dic[line_state[i]]):
                            B_dic[line_state[i]][word_list[i]] = 0.0
                        else:
                            B_dic[line_state[i]][word_list[i]] += 1   #用于计算发射概率
        Output()


### test 的函数
def load_model(f_name):
    with open(f_name,"rb") as ifp:
        return eval(ifp.read())  #eval参数是一个字符串, 可以把这个字符串当成表达式来求值,


def viterbi(obs, states, start_p, trans_p, emit_p):  #维特比算法（一种递归算法）
    V = [{}]
    path = {}
    for y in states:   #初始值
        V[0][y] = start_p[y] * emit_p[y].get(obs[0],0)   #在位置0，以y状态为末尾的状态序列的最大概率
        path[y] = [y]
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:      #从y0 -> y状态的递归
            #(prob, state) = max(([(V[t-1][y0] * trans_p[y0].get(y,0) * emit_p[y].get(obs[t],0) ,y0) for y0 in states if V[t-1][y0]>0])) 
            (prob, state) = max([(V[t-1][y0] * trans_p[y0].get(y,0) * emit_p[y].get(obs[t],0) ,y0) for y0 in states])

            V[t][y] =prob
            newpath[y] = path[state] + [y]
        path = newpath  #记录状态序列
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  #在最后一个位置，以y状态为末尾的状态序列的最大概率
    return prob, path[state]  #返回概率和状态序列


def cut(sentence):
    prob, pos_list =  viterbi(sentence,('B','M','E','S'), prob_start, prob_trans, prob_emit)
    return prob,pos_list

def fenci(test_str,pos_list):
    out_str = ''
    for i in range(len(pos_list)):
        if pos_list[i] == 'B':
            out_str = out_str + test_str[i]
        elif pos_list[i] == 'M':
            out_str = out_str + test_str[i]
        elif pos_list[i] == 'E':
            out_str = out_str + test_str[i] + ' '
        elif pos_list[i] == 'S':
            out_str = out_str + test_str[i] + ' '
    return out_str

if __name__ == "__main__":
    if para == "train":
        train()
    elif para == "test":
        prob_start = load_model("trainHMM\prob_start.py")
        prob_trans = load_model("trainHMM\prob_trans.py")
        prob_emit = load_model("trainHMM\prob_emit.py")
        out_str = ''
        
        
        # 读文件
        with open("CTBtestingset.txt") as f_in:
            with open("output.txt","w") as f_out:
                lines = f_in.readlines()
                for line in lines:
                    if line.startswith("\n"):
                        continue
                
                    else:
                        test_str =  line.split("\n")[0]
                        prob,pos_list = cut(test_str)
                        out_str =  fenci(test_str,pos_list) + '\n'
                        f_out.write(out_str)
