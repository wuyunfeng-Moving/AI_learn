import tensorflow as tf
from tensorflow.keras import layers, models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 创建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 输入层
model.add(layers.MaxPooling2D((2, 2)))  # 池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 卷积层
model.add(layers.MaxPooling2D((2, 2)))  # 池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 卷积层
model.add(layers.Flatten())  # 展平层
model.add(layers.Dense(64, activation='relu'))  # 全连接层
model.add(layers.Dense(10))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 加载数据集（以CIFAR-10为例）
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0  # 数据归一化
test_images = test_images.astype('float32') / 255.0

# 训练模型
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# 添加图片预测功能
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image

def predict_image():
    # 打开文件选择对话框
    print("请选择图片")
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title='选择图片',
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')]
    )
    
    if file_path:
        # 加载并预处理图片
        img = Image.open(file_path)
        img = img.resize((32, 32))  # 调整为模型输入大小
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度

        # 预测
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        
        # CIFAR-10类别名称
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f'预测结果: {class_names[predicted_class]}')

# 运行预测
predict_image()

