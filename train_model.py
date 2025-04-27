import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
df=pd.read_csv(r'C:\Users\SHIVALI SONI\Downloads\py-master\ML\2_linear_reg_multivariate\salary_predictor_project\Model\hiring.csv')
df
import math
median_testscore=math.floor(df['test_score(out of 10)'].median())
median_testscore   
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(median_testscore)
df['experience'] = df['experience'].fillna('zero')
df['experience'] = df['experience'].apply(lambda x: w2n.word_to_num(x))
df
reg=linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],df['salary($)'])
import pickle
# Save model
pickle.dump(reg, open('salary_predictor.pkl', 'wb'))
