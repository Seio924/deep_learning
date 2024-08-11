import tensorboard as tf
import numpy as np

Pmodel = tf.keras.load_model('C:\GitHub\deep_learning\Project3\model1')

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

first_input = changed_to_number[117:117+25]
first_input = tf.one_hot(first_input, 31)
first_input = tf.expand_dims(first_input, axis=0)


#연속 예측
music = []

for i in range(200):

    predict_data = Pmodel.predict(first_input)

    #가장 높은 확률을 가진 것으로 예측값 생성
    predict_data = np.argmax(predict_data[0])
    
    #랜덤으로 예측값 생성 (반복 때문에)
    #predict_data = np.random.choice(unic_text, 1, p=predict_data[0])

    music.append(predict_data)

    next_input = first_input.numpy()[0][1:]

    one_hot_num = tf.one_hot(predict_data, 31)

    first_input = np.vstack([next_input, one_hot_num.numpy()])
    first_input = tf.expand_dims(first_input, axis=0)

music_text = []

for i in music:
    music_text.append(num_to_text[i])
