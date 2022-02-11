#import libraries
import pandas as pd
import numpy as np

#missing values are question marks in this data set
missingValues = ["?"]


#read csv into dataframe 
df = pd.read_csv("SkillCraft1_Dataset.csv", na_values = missingValues)

#look at first few rows
print(df.tail())

#look at Age column
print(df['Age'])
print(df['Age'].isnull())
print("*****************************************************")


#look at sum of 
print(df['Age'])
print(df['Age'].isnull())
print("*****************************************************")

#total missing value stats
print("Missing values stats:\n\n")
print(df.isnull().sum())

print("*****************************************************")


#total number of missing values
print("Total missing values: ",df.isnull().sum().sum())
print("\n")

print("*****************************************************")

print(df.info())

print("*****************************************************")

print(df.describe())

print("*****************************************************")

print(df.columns)

print("*****************************************************")


print(df.dtypes)

print("*****************************************************")

print("Dropping columns 'GameID','Age' , 'HoursPlayed', 'TotalHours'")

truncated_df = df.drop(["Age", "HoursPerWeek", "TotalHours"], axis=1)

print("\n",truncated_df.head())

print("*****************************************************")

print("Outputting to file...")

truncated_df.to_csv("SkillCraft1_Dataset_modified_withGID.csv", index=False, encoding='utf8')
