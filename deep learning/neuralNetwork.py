import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape input data
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.keras.Input(shape=(784,)) # input images
y = tf.keras.Input(shape=(n_classes,)) # labels


class NeuralNetworkModel(tf.keras.Model):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.hidden_1_layer = tf.keras.layers.Dense(n_nodes_hl1, activation='relu')
        self.hidden_2_layer = tf.keras.layers.Dense(n_nodes_hl2, activation='relu')
        self.hidden_3_layer = tf.keras.layers.Dense(n_nodes_hl3, activation='relu')
        self.output_layer = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x = self.hidden_1_layer(inputs)
        x = self.hidden_2_layer(x)
        x = self.hidden_3_layer(x)
        x = self.output_layer(x)
        return x


def train_neural_network(x):
    y_train_one_hot = to_categorical(y_train, n_classes)
    y_test_one_hot = to_categorical(y_test, n_classes)

    model = NeuralNetworkModel()

    prediction = model(x)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y, prediction)

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    hm_epochs = 10
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for i in range(int(len(x_train) / batch_size)):
            start = i * batch_size
            end = start + batch_size
            epoch_x, epoch_y = x_train[start:end], y_train_one_hot[start:end]

            with tf.GradientTape() as tape:
                prediction = model(epoch_x)
                loss_value = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(epoch_y, prediction)

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss_value
        print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss}")

train_neural_network(x)
