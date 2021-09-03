import pandas as pd


mma_data = pd.read_csv('mma_data.csv', index_col = 0)
print(mma_data)
# results = [0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,-1,1,1,1,1,-1,1,0,0,0,-1,-1,0,-1,0,0,0,1,-1,0,1]
# print(len(results) == len(mma_data))

