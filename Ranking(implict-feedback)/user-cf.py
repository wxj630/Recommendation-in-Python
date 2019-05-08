import math
import numpy as np
import pandas as pd
from collections import defaultdict # defaultdict

#
def UserSimilarity(train):
    # build inverse table for item_user
    item_user = dict()
    for u,items in train.items():
        for i in items.keys():
            if i not in item_user:
                item_user[i] = set()
            item_user[i].add(u)
    print('item_user对二维字典为：')
    print(item_user)
    print('\n')

    # calculated co-rated items between users
    C = defaultdict(defaultdict)
    N = defaultdict(int)
    for i,users in item_user.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                # C[u][v] += 1
                C[u][v] += 1/ math.log(1+len(users)) # 惩罚热门物品
    print('user出现次数矩阵：')
    print(np.asarray(list(N)))
    print('user共现矩阵：')
    print(pd.DataFrame(C))
    print('\n')\

    # calculate final similarity matrix W
    W = dict()
    for u,related_users in C.items():
        for v,cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])


    return W

def Recommend(user,train,W,K):
    rank = dict()
    interacted_items = train[user]
    for v,wuv in sorted(W[user].items(),key=lambda items:items[1],reverse=True)[0:K]:
        for i,rvi in train[v].items():
            if i in interacted_items: # 过滤掉已经买的
                continue
            rank[i] += wuv * rvi
    return rank

if __name__ == '__main__':
    Train_Data = {'A': {'i1': 1, 'i2': 1, 'i4': 1},
                  'B': {'i1': 1, 'i4': 1},
                  'C': {'i1': 1, 'i2': 1, 'i5': 1},
                  'D': {'i2': 1, 'i3': 1},
                  'E': {'i3': 1, 'i5': 1},
                  'F': {'i2': 1, 'i4': 1}
        }
    W= UserSimilarity(Train_Data)
    print('用户相似度二维字典W：')
    print(W)
    print('用户相似度矩阵化:')
    print(pd.DataFrame(W))
    # print('推荐得分为（排序）：')
    # R = Recommend(Train_Data, 'C', W, 5) # 为用户C推荐5个物品
    # print(sorted(R.items(),key=lambda items:items[1],reverse=True))
    # print(sorted(R.items(), key=lambda items: items[1], reverse=True)[0]) # 最优先推荐