import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns


#def pairPlot(df):
#	from matplotlib import pyplot as plt
#    from matplotlib import cm as cm

#   sns.pairplot(df)

#	plt.show()

def correlationMatrix(df):
	from matplotlib import pyplot as plt

	corr = df.corr()
	#Plot figsize
	fig, ax = plt.subplots(figsize=(10, 10))
	#Generate Color Map, red & blue
	colormap = sns.diverging_palette(220, 10, as_cmap=True)
	#Generate Heat Map, allow annotations and place floats in map
	sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
	#Apply xticks
	plt.xticks(range(len(corr.columns)), corr.columns);
	#Apply yticks
	plt.yticks(range(len(corr.columns)), corr.columns)

	plt.title('SkillCraft1 Feature Correlation')
	#show plot



	# Create Correlation df
	#corr = df.corr()
	# Plot figsize
	#fig, ax = plt.subplots(figsize=(10, 10))
	# Generate Color Map
	#colormap = sns.diverging_palette(220, 10, as_cmap=True)

	# Drop self-correlations
	#dropSelf = np.zeros_like(corr)
	#dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map
	#colormap = sns.diverging_palette(220, 10, as_cmap=True)
	# Generate Heat Map, allow annotations and place floats in map
	#sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
	# Apply xticks
	#plt.xticks(range(len(corr.columns)), corr.columns);
	# Apply yticks
	#plt.yticks(range(len(corr.columns)), corr.columns)
	#show plot

	plt.show()

#missing values are question marks in this data set
missingValues = ["?"]

#read csv into dataframe 
df = pd.read_csv("SkillCraft1_Dataset_modified.csv", na_values = missingValues)

print(df.head())
print(df.describe())


#pairPlot(df)
correlationMatrix(df)


