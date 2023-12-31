import pandas as pd

#csv 파일 읽기
data = pd.read_csv('./gpascore.csv')

# 0. 데이터 전처리 하기

#학습데이터 중 없는 데이터 (빈공간) 개수 출력
#print(data.isnull().sum())

#빈공간이 있는 행 삭제
data = data.dropna()

#빈공간을 채워줌
#data = data.fillna(100)

#gre열의 최솟값 출력 [min, max, count]
#print(data['gre'].min())

pred_data = data['admit'].values #admit열 데이터 리스트에 담기

train_data = []

#iterrows : 한 행씩 출력
for i, rows in data.iterrows():
    train_data.append([rows['gre'], rows['gpa'], rows['rank']])    


import numpy as np
import tensorflow as tf

# 1. 딥러닝 model 디자인하기

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), #레이어 | 괄호 안 숫자 : 레이어 하나 당 node 개수 | 레이어에 activation function 넣기
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'), #예측하고싶은 결과의 수 만큼 node 개수, 원하는 결과 값이 0~1 사이이므로 sigmoid 사용 
]) #신경망 레이어들 쉽게 만들어줌


# 2. model compile하기

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#optimizer : 경사하강법으로 w 변화시킬때 더 빨리 예측할수있도록 w를 얼마나 변화시킬건지에 관해 정해주는 것
#loss : 오차 검출 함수


# 3. model 학습(fit) 시키기

#model.fit(학습데이터, 예측데이터, 몇 번 학습을 시킬 지)
model.fit(np.array(train_data), np.array(pred_data), epochs=1000)


# 4. 예측하기

predict_result = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ] )