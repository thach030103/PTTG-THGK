import os
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Hàm nén ảnh bằng phương pháp SVD
def compress_svd(img, rank):
    U, S, VT = np.linalg.svd(img, full_matrices=False)
    U_rank = U[:, :rank]
    S_rank = np.diag(S[:rank])
    VT_rank = VT[:rank, :]
    compressed_img = U_rank @ S_rank @ VT_rank
    return compressed_img

# Đường dẫn đến ảnh
img_path = os.path.join('img', 'dog.jpg')

# Đọc ảnh từ đường dẫn
img = imread(img_path)

# Chuyển ảnh màu thành ảnh grayscale
img_grayscale = np.mean(img, -1)

# Kích thước của ảnh
n_rows, n_cols = img_grayscale.shape

print(f'Original image: {img.shape}')
print(f'Grayscaled image: {img_grayscale.shape}')

# Nén ảnh với rank được chọn
rank = 10
compressed_img = compress_svd(img_grayscale, rank)

# Hiển thị ảnh gốc và ảnh nén
figure = plt.figure(figsize=(12, 6))

img1 = figure.add_subplot(1, 2, 1)
img1.set_title('Original image')
img1.imshow(img_grayscale, cmap='gray')

img2 = figure.add_subplot(1, 2, 2)
img2.set_title(f'Compressed image (Rank= {rank})')
img2.imshow(compressed_img, cmap='gray')

plt.show()
