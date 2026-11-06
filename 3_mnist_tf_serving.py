"""Train a small MNIST neural network in TensorFlow, export it, serve it with TF-Serving (Docker), and query it.

Step 1: python 3_mnist_tf_serving.py train        -> trains, prints test accuracy, saves models/mnist/1
Step 2: run TF-Serving in Docker (from this folder):
   docker run -p 8501:8501 --mount type=bind,source=$(pwd)/models/mnist,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving
   (Windows PowerShell: replace $(pwd) with ${PWD})
Step 3: python 3_mnist_tf_serving.py query         -> sends test images to the REST endpoint, prints accuracy
"""
import sys, json
import numpy as np

def train():
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    xtr, xte = xtr / 255.0, xte / 255.0
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dense(10)])
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(xtr, ytr, epochs=5, validation_split=0.1)
    loss, acc = model.evaluate(xte, yte)
    print("Test accuracy:", acc)
    model.export("models/mnist/1")   # SavedModel with version folder "1"

def query():
    import requests, tensorflow as tf
    (_, _), (xte, yte) = tf.keras.datasets.mnist.load_data()
    xte = xte[:200] / 255.0
    r = requests.post("http://localhost:8501/v1/models/mnist:predict", data=json.dumps({"instances": xte.tolist()}))
    r.raise_for_status()
    preds = np.argmax(np.array(r.json()["predictions"]), axis=1)
    print("Served accuracy on 200 test images:", (preds == yte[:200]).mean())

if __name__ == "__main__":
    {"train": train, "query": query}[sys.argv[1]]()
