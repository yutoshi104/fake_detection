# import tensorflow as tf
# # from keras.backend.tensorflow_backend import set_session
# # config = tf.ConfigProto(
# #     gpu_options=tf.GPUOptions(
# #         visible_device_list="2", # specify GPU number
# #         allow_growth=True
# #     )
# # )
# # set_session(tf.Session(config=config))


# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# exit()


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
