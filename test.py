import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

img_align_celeba = './img_align_celeba/'
data_1000_img = './data_10_img/'




# Tải dữ liệu celeba
def load_celeba_data(img_dir, img_size=(178, 218)):
    images = []
    for img_name in os.listdir(img_dir)[:1000]:  # Giới hạn 1000 ảnh để dễ dàng thử nghiệm
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).resize(img_size)
        images.append(np.array(img))
    images = np.array(images)
    return images / 255.0


celeba_images = load_celeba_data(data_1000_img)
input_shape = celeba_images.shape[1:]

# Chọn ra 10 ảnh từ celeba_images để hiển thị
sample_images = celeba_images[:10]

# Tạo figure với 2 hàng và 5 cột để hiển thị 10 ảnh
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# Định nghĩa watermark cố định
watermark_size = 32
fixed_watermark = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])


# Giai đoạn 1: Huấn luyện Mạng Watermarking
def build_encoder(input_shape, watermark_size):
    inputs = layers.Input(shape=input_shape)
    wm_inputs = layers.Input(shape=(watermark_size,))
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    wm_dense = layers.Dense(np.prod(x.shape[1:]), activation='relu')(wm_inputs)
    wm_reshaped = layers.Reshape(x.shape[1:])(wm_dense)
    
    x = layers.Concatenate()([x, wm_reshaped])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model([inputs, wm_inputs], encoded, name="encoder")


def build_decoder(input_shape, watermark_size):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    decoded = layers.Dense(watermark_size, activation='sigmoid')(x)
    
    return models.Model(inputs, decoded, name="decoder")
