import math
import pandas as pd
import numpy as np
## ItemCF-余弦算法,alpha参数用于惩罚热门物品，N[i]越大越热门
def ItemSimilarity_cos(train,alpha):
    C = dict()   ##书本对同时被购买的次数
    N = dict()   ##书本被购买用户数
    for u,items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i]=0
            N[i] += items[i]* items[i]
            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i]=dict()
                if j not in C[i].keys():
                    C[i][j]=0
                ##当用户同时购买了i和j，则加评分乘积
                C[i][j] += items[i]*items[j]
    print('各物品被评分的平方N:')
    print(np.asarray(list(N.items())))
    print("同时被购买的评分乘积C:")
    print(pd.DataFrame(C))

    W = dict()  ##书本对相似分数
    for i,related_items in C.items():
        if i not in W.keys():
            W[i]=dict()
        for j,cij in related_items.items():
            W[i][j] = cij / (math.pow( N[i],alpha) * math.pow( N[j],1-alpha) )
    return W

# 结合用户喜好对物品排序
def Recommend(train,user_id,W,K):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        tmp=W[i]
        for j,wj in sorted(tmp.items(),key=lambda d: d[1],reverse=True)[0:K]:
            if j not in rank.keys():
                rank[j]=0
            ##r如果用户已经购买过，则不再推荐
            if j in ru:
                continue
            ##待推荐的书本j与用户已购买的书本i相似，则累加上相似分数
            rank[j] += pi*wj
    return rank

if __name__ == '__main__':
    Train_Data = {'A':{'i1':5,'i2':5 ,'i4':4},
     'B':{'i1':3,'i4':4},
     'C':{'i1':4,'i2':4,'i5':5},
     'D':{'i2':4,'i3':1},
     'E':{'i3':1,'i5':5},
     'F':{'i2':1,'i4':5}
        }
    W= ItemSimilarity_cos (Train_Data,0.3)
    print('物品相似度二维字典W：')
    print(W)
    print('物品相似度矩阵化:')
    print(pd.DataFrame(W))
    print('推荐得分为（排序）：')
    R = Recommend(Train_Data, 'C', W, 5) # 为用户C推荐5个物品
    print(sorted(R.items(),key=lambda items:items[1],reverse=True))
    print(sorted(R.items(), key=lambda items: items[1], reverse=True)[0]) # 最优先推荐
