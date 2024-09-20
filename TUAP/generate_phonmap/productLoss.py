import numpy as np
import Levenshtein
from g2p_en import G2p
from weighted_levenshtein import lev, osa, dam_lev

g2p = G2p()

def product(df, modifiedlist, targetlist):
    # if len(modifiedlist) > len(targetlist):
    #     modifiedlist = modifiedlist[0:len(targetlist)]
    # print(modifiedlist,targetlist)
    newlist = []
    for i,ch in enumerate(targetlist):
        if ch == 'AA0' or ch == 'AA1' or ch == 'AA2' or ch == 'AA':
            if i<len(targetlist)-1 and targetlist[i+1] == 'R':
                newlist.append('ER')
                continue
            else:
                newlist.append('AA')
        elif ch == 'R':
            if targetlist[i-1] == 'AA0' or targetlist[i-1] == 'AA1' or targetlist[i-1] == 'AA2' or targetlist[i-1] == 'AA':
                continue
            else:
                newlist.append('R')
        elif ch == 'AE0' or ch == 'AE1' or ch == 'AE2' or ch == 'AE':
            newlist.append('AE')
        elif ch == 'AH1' or ch == 'AH2' or ch == 'AH':
            newlist.append('AH')
        elif ch == 'AO0' or ch == 'AO1' or ch == 'AO2' or ch == 'AO':
            newlist.append('AO')
        elif ch == 'AW0' or ch == 'AW1' or ch == 'AW2' or ch == 'AW':
            newlist.append('AW')
        elif ch == 'AY0' or ch == 'AY1' or ch == 'AY2' or ch == 'AY':
            newlist.append('AY')
        elif ch == 'EH0' or ch == 'EH1' or ch == 'EH2' or ch == 'EH':
            newlist.append('EH')
        elif ch == 'ER0' or ch == 'ER1' or ch == 'ER2' or ch == 'ER':
            newlist.append('ER')
        elif ch == 'EY0' or ch == 'EY1' or ch == 'EY2' or ch == 'EY':
            newlist.append('EY')
        elif ch == 'HH0' or ch == 'HH1' or ch == 'HH2' or ch == 'HH':
            newlist.append('HH')
        elif ch == 'IH0' or ch == 'IH1' or ch == 'IH2' or ch == 'IH':
            newlist.append('IH')
        elif ch == 'IY0' or ch == 'IY1' or ch == 'IY2' or ch == 'IY':
            newlist.append('IY')
        elif ch == 'OW0' or ch == 'OW1' or ch == 'OW2' or ch == 'OW':
            newlist.append('OW')
        elif ch == 'OY0' or ch == 'OY1' or ch == 'OY2' or ch == 'OY':
            newlist.append('OY')
        elif ch == 'UH0' or ch == 'UH1' or ch == 'UH2' or ch == 'UH':
            newlist.append('UH')
        elif ch == 'UW0' or ch == 'UW1' or ch == 'UW2' or ch == 'UW':
            newlist.append('UW')
        elif ch == '\'' or ch == ' ' :  #
            continue
        else:
            newlist.append(ch)
    newlist1 = []
    for i,ch in enumerate(modifiedlist):
        if ch == 'AA0' or ch == 'AA1' or ch == 'AA2' or ch == 'AA':
            if i<len(modifiedlist)-1 and modifiedlist[i + 1] == 'R':
                newlist1.append('ER')
                continue
            else:
                newlist1.append('AA')
        elif ch == 'R':
            if modifiedlist[i - 1] == 'AA0' or modifiedlist[i - 1] == 'AA1' or modifiedlist[i - 1] == 'AA2' or modifiedlist[
                i - 1] == 'AA':
                continue
            else:
                newlist1.append('R')
        elif ch == 'AE0' or ch == 'AE1' or ch == 'AE2' or ch == 'AE':
            newlist1.append('AE')
        elif ch == 'AH1' or ch == 'AH2' or ch == 'AH':
            newlist1.append('AH')
        elif ch == 'AO0' or ch == 'AO1' or ch == 'AO2' or ch == 'AO':
            newlist1.append('AO')
        elif ch == 'AW0' or ch == 'AW1' or ch == 'AW2' or ch == 'AW':
            newlist1.append('AW')
        elif ch == 'AY0' or ch == 'AY1' or ch == 'AY2' or ch == 'AY':
            newlist1.append('AY')
        elif ch == 'EH0' or ch == 'EH1' or ch == 'EH2' or ch == 'EH':
            newlist1.append('EH')
        elif ch == 'ER0' or ch == 'ER1' or ch == 'ER2' or ch == 'ER':
            newlist1.append('ER')
        elif ch == 'EY0' or ch == 'EY1' or ch == 'EY2' or ch == 'EY':
            newlist1.append('EY')
        elif ch == 'HH0' or ch == 'HH1' or ch == 'HH2' or ch == 'HH':
            newlist1.append('HH')
        elif ch == 'IH0' or ch == 'IH1' or ch == 'IH2' or ch == 'IH':
            newlist1.append('IH')
        elif ch == 'IY0' or ch == 'IY1' or ch == 'IY2' or ch == 'IY':
            newlist1.append('IY')
        elif ch == 'OW0' or ch == 'OW1' or ch == 'OW2' or ch == 'OW':
            newlist1.append('OW')
        elif ch == 'OY0' or ch == 'OY1' or ch == 'OY2' or ch == 'OY':
            newlist1.append('OY')
        elif ch == 'UH0' or ch == 'UH1' or ch == 'UH2' or ch == 'UH':
            newlist1.append('UH')
        elif ch == 'UW0' or ch == 'UW1' or ch == 'UW2' or ch == 'UW':
            newlist1.append('UW')
        elif ch == '\''or ch == ' ': #
            continue
        else:
            newlist1.append(ch)

# AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN
# 'AE1', 'Z', ' ', 'F', 'AO1', 'R', ' ', 'EH1', 'CH', 'IH0', 'NG', 'Z', ' ', 'DH', 'EY1', ' ', 'AA1', 'R', ' ', 'AH1', 'V', ' ', 'T', 'UW1', ' ', 'K', 'A
# Y1', 'N', 'D', 'Z', ' ', 'B', 'R', 'IH1', 'T', 'IH0', 'SH', ' ', 'AH0', 'N', 'D', ' ', 'F', 'AO1', 'R', 'AH0', 'N'
# POWER OFF
# THEY ARE OF
# P  AW1  ER0  AO2 F /ˈpaʊər//ɔːf/
# DH EY1  AA1,R  AH1,V
# 1  0  固定组合0 连着等于0的     空格怎么用？？？
#两个音素数量不一样时，找到最连续、最小的target音素数量长度的斜线定位
#当到一个阈值（长度一样）时，定位不一样的音素加权
    # for i, j in zip(newlist1, newlist):
    #     if i == j :
    #         sum += 1
    #         continue
    #     if j not in df[i].keys():
    #         sum += df[j][i]
    #         continue
    #     sum += df[i][j]
    sumlist =  []
    sum = 0

    minlist=[0 for i in range(max(len(newlist1)-len(newlist)+1,1))]
    # print(len(minlist))

    for i,cht in enumerate(newlist): #对target每个音素过一遍所有叠加音频的音素
        # if i == ' ':
        #     sumlist.append(sum)
        #     sum = 0
        #     print(minlist)
        #     minlist = []
        #     continue
        # minsum = df[i][newlist1[0]]
        for j in range(len(minlist)):

            # if df[i][j]<minsum:
            #     minsum=df[i][j]
            #     minlist.append(j)
            # sum += df[i][j+i]
            if i<len(newlist1):

                minlist[j]+= df[cht][newlist1[i+j]] #max7min0
            else :
                minlist[j]+= 7
    # print(min(minlist)/ len(newlist)) #除target的音素list长度
    # for k in range(len(newlist)):
    #     print(newlist1[minlist.index(min(minlist))+k])

    return (min(minlist) / len(newlist))

def convert(targetlist,modifiedlist):
    newlist = []
    for i,ch in enumerate(targetlist):
        if ch == 'AA0' or ch == 'AA1' or ch == 'AA2' or ch == 'AA':
            if i<len(targetlist)-1 and targetlist[i+1] == 'R':
                newlist.append('ER')
                continue
            else:
                newlist.append('AA')
        elif ch == 'R':
            if targetlist[i-1] == 'AA0' or targetlist[i-1] == 'AA1' or targetlist[i-1] == 'AA2' or targetlist[i-1] == 'AA':
                continue
            else:
                newlist.append('R')
        elif ch == 'AE0' or ch == 'AE1' or ch == 'AE2' or ch == 'AE':
            newlist.append('AE')
        elif ch == 'AH1' or ch == 'AH2' or ch == 'AH':
            newlist.append('AH')
        elif ch == 'AO0' or ch == 'AO1' or ch == 'AO2' or ch == 'AO':
            newlist.append('AO')
        elif ch == 'AW0' or ch == 'AW1' or ch == 'AW2' or ch == 'AW':
            newlist.append('AW')
        elif ch == 'AY0' or ch == 'AY1' or ch == 'AY2' or ch == 'AY':
            newlist.append('AY')
        elif ch == 'EH0' or ch == 'EH1' or ch == 'EH2' or ch == 'EH':
            newlist.append('EH')
        elif ch == 'ER0' or ch == 'ER1' or ch == 'ER2' or ch == 'ER':
            newlist.append('ER')
        elif ch == 'EY0' or ch == 'EY1' or ch == 'EY2' or ch == 'EY':
            newlist.append('EY')
        elif ch == 'HH0' or ch == 'HH1' or ch == 'HH2' or ch == 'HH':
            newlist.append('HH')
        elif ch == 'IH0' or ch == 'IH1' or ch == 'IH2' or ch == 'IH':
            newlist.append('IH')
        elif ch == 'IY0' or ch == 'IY1' or ch == 'IY2' or ch == 'IY':
            newlist.append('IY')
        elif ch == 'OW0' or ch == 'OW1' or ch == 'OW2' or ch == 'OW':
            newlist.append('OW')
        elif ch == 'OY0' or ch == 'OY1' or ch == 'OY2' or ch == 'OY':
            newlist.append('OY')
        elif ch == 'UH0' or ch == 'UH1' or ch == 'UH2' or ch == 'UH':
            newlist.append('UH')
        elif ch == 'UW0' or ch == 'UW1' or ch == 'UW2' or ch == 'UW':
            newlist.append('UW')
        elif ch == '\'' or ch == ' ' :  #
            continue
        else:
            newlist.append(ch)
    newlist1 = []
    for i,ch in enumerate(modifiedlist):
        if ch == 'AA0' or ch == 'AA1' or ch == 'AA2' or ch == 'AA':
            if i<len(modifiedlist)-1 and modifiedlist[i + 1] == 'R':
                newlist1.append('ER')
                continue
            else:
                newlist1.append('AA')
        elif ch == 'R':
            if modifiedlist[i - 1] == 'AA0' or modifiedlist[i - 1] == 'AA1' or modifiedlist[i - 1] == 'AA2' or modifiedlist[
                i - 1] == 'AA':
                continue
            else:
                newlist1.append('R')
        elif ch == 'AE0' or ch == 'AE1' or ch == 'AE2' or ch == 'AE':
            newlist1.append('AE')
        elif ch == 'AH1' or ch == 'AH2' or ch == 'AH':
            newlist1.append('AH')
        elif ch == 'AO0' or ch == 'AO1' or ch == 'AO2' or ch == 'AO':
            newlist1.append('AO')
        elif ch == 'AW0' or ch == 'AW1' or ch == 'AW2' or ch == 'AW':
            newlist1.append('AW')
        elif ch == 'AY0' or ch == 'AY1' or ch == 'AY2' or ch == 'AY':
            newlist1.append('AY')
        elif ch == 'EH0' or ch == 'EH1' or ch == 'EH2' or ch == 'EH':
            newlist1.append('EH')
        elif ch == 'ER0' or ch == 'ER1' or ch == 'ER2' or ch == 'ER':
            newlist1.append('ER')
        elif ch == 'EY0' or ch == 'EY1' or ch == 'EY2' or ch == 'EY':
            newlist1.append('EY')
        elif ch == 'HH0' or ch == 'HH1' or ch == 'HH2' or ch == 'HH':
            newlist1.append('HH')
        elif ch == 'IH0' or ch == 'IH1' or ch == 'IH2' or ch == 'IH':
            newlist1.append('IH')
        elif ch == 'IY0' or ch == 'IY1' or ch == 'IY2' or ch == 'IY':
            newlist1.append('IY')
        elif ch == 'OW0' or ch == 'OW1' or ch == 'OW2' or ch == 'OW':
            newlist1.append('OW')
        elif ch == 'OY0' or ch == 'OY1' or ch == 'OY2' or ch == 'OY':
            newlist1.append('OY')
        elif ch == 'UH0' or ch == 'UH1' or ch == 'UH2' or ch == 'UH':
            newlist1.append('UH')
        elif ch == 'UW0' or ch == 'UW1' or ch == 'UW2' or ch == 'UW':
            newlist1.append('UW')
        elif ch == '\''or ch == ' ': #
            continue
        else:
            newlist1.append(ch)
    return newlist,newlist1

def insert(a):
#这里定义插入数据a时的权重
#比如插入数据是很重要的，权重设为10
#我没有定义删除数据a时的函数，因为我假设删除和添加的权重是一样的。
    return 1
 
def subtitute(map,a,b):
#这里定义将数据a修改为数据b时的权重
#假设设置为a与b的差
    return map[a][b]/7
 
def weightedDistance(map, targetlist: list, modifiedlist: list) -> int:
    '''
        加权动态规划求解
    '''
    target,modified=convert(targetlist,modifiedlist)
    # print(len(Levenshtein.editops(modifiedlist,targetlist)),len(Levenshtein.editops(modified,target)),Levenshtein.editops(modified,target))
    m = len(modified)#target
    n = len(target)#modified
    # print(modified,target)
 
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
 
    dp[0][0] = 0
    #要记得初始化权重，在这里就已经要加权计算了
    for i in range(1,m+1):
        dp[i][0] = dp[i-1][0] + insert(modified[i-1])
    for j in range(1,n+1):
        dp[0][j] = dp[0][j-1] + insert(target[j-1])
    
    # for i in range(m+1):
    #     for j in range(n+1):
    #         print(dp[i][j],end=' ')
    #     print()
    for i in range(1,1+m):
        for j in range(1,1+n):
            if modified[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                #对三种情况分别计算，再比较
                t1=dp[i-1][j-1]+subtitute(map,target[j-1],modified[i-1])               
                t2=dp[i][j-1]+insert(target[j-1])
                t3=dp[i-1][j]+insert(modified[i-1])#delete
                dp[i][j] = min(t1,t2,t3)
                # print(t1,t2,t3,dp[i][j],end="!")
    # for i in range(m+1):
    #     for j in range(n+1):
    #          print(dp[i][j],modified[i-1],target[j-1],end=' ')
    #     print()
    return dp[m][n]

if __name__ == '__main__':
    phone_map = np.load('./phone_map.npy', allow_pickle=True)
    phone_map = dict(phone_map)
    # product(phone_map,g2p("POWER "),g2p("POWER OFF"))
    print(weightedDistance(phone_map,['P', 'OW', 'OW', 'ER', 'F'],['P', 'OW', 'B', 'ER', 'AO', 'F']))
    
    # transpose_costs = np.ones((128, 128), dtype=np.float64)
    # transpose_costs[97, 98] = 0.75  #ord( make swapping 'A' for 'B' cost 0.75

# note: now using dam_lev. lev does not support swapping, but osa and dam_lev do.
# See Wikipedia links for difference between osa and dam_lev
   #print(dam_lev(bytes(['P', 'IY']).encode("utf-8"),bytes(['P', 'OW']).encode("utf-8"), transpose_costs=transpose_costs))  # prints '0.75'
