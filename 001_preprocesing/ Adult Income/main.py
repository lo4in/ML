import pandas as pd
import matplotlib.pyplot as plt

path = "netflix/std_data.csv"

df = pd.read_csv(path)




print(df[['StudyTimeWeekly', 'GPA']])

plt.scatter(df['StudyTimeWeekly'], df['GPA'])
plt.xlabel('StudyTimeWeekly')
plt.ylabel('GPA')

plt.show()