import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('./cmudict.xlsx',header=None)
phone_map = {}
map_array =[[0 for i in range(40)] for j in range(40)]
for ind,nam in enumerate(df.iloc[32]):
    phone_map[nam] = {}
    phone_map[nam][nam] = 0
for index,name in enumerate(df.iloc[32]):

    if type(name)==float:continue

    for i in range(len(df.iloc[32]) - 1 - index):
        tmp = 0
        # print(df.iloc[20][index+i])
        for j in range(11):

            if np.isnan(df.iloc[20 + j][index]):
                df.iloc[20 + j][index] = 0
            if np.isnan(df.iloc[20 + j][i + index+1]):
                df.iloc[20 + j][i + index+1] = 0
            # print(df.iloc[20 + j][index], df.iloc[20 + j][i + index])
            tmp += abs(df.iloc[20 + j][index] - df.iloc[20 + j][i + index+1])
        #         # print(df.iloc[20 +j][k],df.iloc[20 +j][1 + k])
        # print(tmp)# print(df.iloc[30 ][0],index-1,i + index)# print(df.iloc[20 ][index ],df.iloc[20 ][1 + index ])
        # print(index,i+index)
        map_array[index-1][index+i]=tmp
        map_array[index+i][index-1]=tmp
        phone_map[df.iloc[32][i+1+index]][name] = tmp
        phone_map[name][df.iloc[32][i+1+index]] = tmp

# print(np.array(list(phone_map.items())))# print(phone_map.values(),map_array)
plt.imshow(np.float32(map_array),cmap='hot',origin='upper')
plt.colorbar()
plt.tight_layout()
plt.savefig('features.jpg')
plt.show()

# for index,name in enumerate(df.iloc[32]):
#     if type(name)==float:continue
#     phone_map[name] = {}
#     for i in range(len(df.iloc[32]) - 1 - index):
#         phone_map[name][df.iloc[32][i+1+index]] = df.iloc[32+index][1+i+index]
# phone_map_array = np.array(list(phone_map.items()))
np.save('phone_map.npy', np.array(list(phone_map.items())))
