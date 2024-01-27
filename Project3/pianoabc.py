import numpy as np
import tensorflow as tf

text = open('C:\GitHub\deep_learning\Project3\pianoabc.txt', 'r').read() 

unic_text = list(set(text))
unic_text.sort()


#utilities
text_to_num = {}
num_to_text = {}

for i, data in enumerate(unic_text):
    text_to_num[data] = i
    num_to_text[i] = data

changed_to_number = []

for i in text:
    changed_to_number.append(text_to_num[i])

trainX = []
trainY = []
i = 0

while i < len(changed_to_number)-25:
    trainX.append(changed_to_number[i:i+25])
    trainY.append(changed_to_number[i+25])
    i += 1

trainX = tf.one_hot(trainX, 31)
trainY = tf.one_hot(trainY, 31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, 31)),
    tf.keras.layers.Dense(31, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=64, epochs=30, verbose=2)

model.save('학습한 모델을 저장시키고 싶은 경로')