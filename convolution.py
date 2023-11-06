import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# Convolution : (AxB) pixel with (3x3) filter => (A-2)x(B-2) pixel
# Pooling : (AxB) pixel with (2x2) filter => (A/2)x(B/2) pixel
# Flatten : (AxB) pixel => (AxB) pixel
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)), #(26 x 26) images after convolution
  tf.keras.layers.MaxPooling2D(2,2), #(13 x 13) images after Pooling
  tf.keras.layers.Conv2D(64,(3,3), activation='relu'), #(11 x 11) images after convolution
  tf.keras.layers.MaxPooling2D(2,2), #(5 x 5) images after convolution
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=50)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

print(model.summary())