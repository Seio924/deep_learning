import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY)= tf.keras.datasets.fashion_mnist.load_data()

#사진 보여주기
#plt.imshow(trainX[0])
#plt.gray()
#plt.colorbar()
#plt.show()

trainX = trainX / 255.0
testX = testX / 255.0


trainX.reshape((trainX.shape[0], 28, 28, 1))
testX.reshape((testX.shape[0], 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#확률 loss 함수 : cross entropy
#relu : 음수 제거 = 0

#sigmoid : 결과를 0~1로 압축 -> binary 예측문제에 사용 [0 또는 1 (대학원 붙는다 or 안붙는다)]
#softmax : 결과를 0~1로 압축 -> 카테고리 예측문제에 사용 [예측한 10개의 확률을 다 더하면 1 나옴]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape="(28, 28, 1)"),
    tf.keras.layers.MaxPoolong2D((2, 2)),
    #tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),
    tf.keras.layers.Flatten(), #일렬로 나열
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=5)
