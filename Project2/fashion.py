import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY)= tf.keras.datasets.fashion_mnist.load_data()

#사진 보여주기
#plt.imshow(trainX[0])




#확률 loss 함수 : cross entropy
#relu : 음수 제거 = 0

#sigmoid : 결과를 0~1로 압축 -> binary 예측문제에 사용 [0 또는 1 (대학원 붙는다 or 안붙는다)]
#softmax : 결과를 0~1로 압축 -> 카테고리 예측문제에 사용 [예측한 10개의 확률을 다 더하면 1 나옴]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])