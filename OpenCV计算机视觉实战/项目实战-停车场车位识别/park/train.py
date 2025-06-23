import os
"""
    Keras 是一个高级神经网络API，可用于快速构建网络模型。现已被集成到TensorFlow中作为其官方高阶API（tf.keras）。
"""
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

"""
    该脚本的主要功能是通过图像数据训练一个基于VGG16的深度学习模型，并将训练最好的一次模型保存到car1.h5文件中。
    
    根据车位被占用和未被占用的图片数据源（该数据源来源于我们上述分析停车场截图中截取出来的占用和未占用的图像）训练模型，
    让模型知道停车位什么情况是占用和未被占用情况。
"""

# 存储训练数据源（图片数据）
files_train = 0
# 存储验证数据源（图片数据）
files_validation = 0

cwd = os.getcwd()
folder = 'train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)

folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)

print(files_train, files_validation)

# 重置图片大小
img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2

model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in model.layers[:10]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),  # lr改为learning_rate
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

checkpoint = ModelCheckpoint(
    "car1.h5",
    monitor='val_accuracy',  # 新版改为val_accuracy
    save_best_only=True,
    mode='auto',
    save_freq='epoch'  # 替换period
)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

history_object = model_final.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,  # 替换samples_per_epoch
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,  # 替换nb_val_samples
    callbacks=[checkpoint, early]
)
