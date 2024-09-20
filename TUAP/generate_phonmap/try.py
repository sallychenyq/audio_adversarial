import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
phonemap = np.load("phone_map.npy",allow_pickle=True)
# print(phonemap)# phonemap = np.(phonemap)# phonemap = dict(phonemap)

similarity=pd.read_excel('cmudict.xlsx').iloc[32:72,1:41].values
print(np.float32(similarity))

plt.imshow(np.float32(similarity),cmap='hot',origin='upper')
plt.colorbar()
plt.tight_layout()
plt.savefig('features.jpg')
plt.show()

