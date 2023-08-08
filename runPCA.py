import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#missing values are question marks in this data set
missingValues = ["?"]

#read csv into dataframe 
df = pd.read_csv("SkillCraft1_Dataset_modified.csv", na_values = missingValues)

df = df.reset_index()

pca = PCA(n_components = 6)
reduced_df = pca.fit(df)

print(reduced_df.head())