import tensorflow as tf
import matplotlib.pyplot as plt


# 读取数据
def load_and_preprocess_data():
    # 划分训练集和测试集
    # 训练数据: 60000张28*28的单通道灰度图
    # 测试数据: 10000张28*28的单通道灰度图
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 验证数据格式，如果是通道在最后的格式，则进行相应的调整
    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    # 将像素值归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # 将目标值转换为分类格式
    # 例如：y_train[0] = 5, 转换为[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape

# 展示数据
def show_data(x_train, y_train):
    # 展示训练集中的前9张图片
    for i in range(9):
        plt.subplot(3, 3, i+1)  # 3行3列
        plt.imshow(x_train[i].reshape(28, 28), cmap='gray')  # 灰度图
        plt.axis('off')  # 不显示坐标轴
        plt.title('label: %d' % y_train[i].argmax(), loc='center') # 显示标签
    plt.show()

# 定义模型
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape), # 32个3*3的卷积核
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), # 2*2的最大池化
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # 64个3*3的卷积核
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), # 2*2的最大池化
        tf.keras.layers.Flatten(),  # 拉平
        tf.keras.layers.Dense(64, activation='relu'),  # 全连接层
        tf.keras.layers.Dense(num_classes, activation='softmax')    # 输出层
    ])
    return model


# 编译模型并训练
def compile_and_train(model, x_train, y_train, x_test, y_test):
    # 编译模型
    model.compile(loss=tf.keras.losses.categorical_crossentropy,    # 损失函数
                  optimizer=tf.keras.optimizers.Adam(),     # 优化器
                  metrics=['accuracy'])     # 评估指标

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    return model

# 评估模型
def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    # 定义超参数
    batch_size = 128   # 批大小
    num_classes = 10    # 分类数
    epochs = 10       # 训练轮数
    img_rows, img_cols = 28, 28     # 图片大小
    (x_train, y_train), (x_test, y_test), input_shape = load_and_preprocess_data()  # 读取数据
    show_data(x_train, y_train)     # 展示数据
    model = create_model(input_shape)   # 创建模型
    model = compile_and_train(model, x_train, y_train, x_test, y_test)  # 编译并训练模型
    evaluate_model(model, x_test, y_test)   # 评估模型
