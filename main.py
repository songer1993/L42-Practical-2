from keras.optimizers import SGD
from keras.utils import np_utils

from data import load_data
from model import get_baseline_model
from model import get_cnn_model

# Load data
print("[INFO] load data...")
(X_train, y_train, X_test, y_test) = load_data()

# Parameters
batch_size = 128
nb_epoch = 100

# Load and compile model
print("[INFO] compiling model...")
model = get_cnn_model()
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0, validation_data=(X_test, y_test))

print("[INFO] evaluating on testing set...")
score = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy:", score[1])