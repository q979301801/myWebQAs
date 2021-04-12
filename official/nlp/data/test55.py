
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑GPU模拟多GPU训练
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

MAX_LEN = 300
BATCH_SIZE = 32
(x_train, y_train), (x_test, y_test) = datasets.reuters.load_data()
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

MAX_WORDS = x_train.max() + 1
CAT_NUM = y_train.max() + 1

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE).cache()

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE).cache()

tf.keras.backend.clear_session()


def create_model():
    model = models.Sequential()

    model.add(layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM, activation="softmax"))
    return (model)


def compile_model(model):
    model.compile(optimizer=optimizers.Nadam(),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.SparseCategoricalAccuracy(), metrics.SparseTopKCategoricalAccuracy(5)])
    return (model)


# 增加以下两行代码
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
    model.summary()
    model = compile_model(model)

history = model.fit(ds_train, validation_data=ds_test, epochs=10)