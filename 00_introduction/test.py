import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def Info():
    print(a.head())
    print("\n")
    print(a.shape)
    print("\n")
    print(a.columns)
    print("\n")
    print(a.info())
    print("\n")
    print(a.describe())
    print(a.describe())
    print(a.mode(numeric_only=False))
    print(a['sex'].value_counts())
    print(a.isnull().sum())
def vis():
    #sns.histplot(a['age'].dropna(), kde=True)
    #plt.title('Распределение возраста')
    #plt.show()

    sns.countplot(data=a, x='sex')
    plt.show() 

a = sns.load_dataset('titanic')

if __name__ == "__main__":
    Info()
    vis()

