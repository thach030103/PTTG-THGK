import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog


# Hàm nén ảnh bằng phương pháp SVD
def svd_compress(img, rank):
    U, S, VT = np.linalg.svd(img, full_matrices=False)
    U_rank = U[:, :rank]
    S_rank = np.diag(S[:rank])
    VT_rank = VT[:rank, :]
    compressed_img = U_rank @ S_rank @ VT_rank
    return compressed_img

def compress_image():
    global img_gray
    global rank_to_compress

    compressed_img = svd_compress(img_gray, rank_to_compress)

    # Đóng figure chứa ảnh gốc
    plt.close(fig_original)

    # Hiển thị ảnh gốc và ảnh nén
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].set_title('Original Image')
    ax[0].imshow(img_gray, cmap='gray')

    ax[1].set_title(f'Compressed Image (Rank={rank_to_compress})')
    ax[1].imshow(compressed_img, cmap='gray')

    # Cập nhật canvas
    plt.show()


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = plt.imread(file_path)
        img_gray = np.mean(img, -1)
        update_image(img_gray)
        

def update_image(img):
    global img_gray
    img_gray = img
    ax.imshow(img_gray, cmap='gray')
    canvas.draw()

# Khởi tạo cửa sổ tkinter
root = tk.Tk()
root.title('Image Compression using SVD')

# Tạo button để mở ảnh
open_button = tk.Button(root, text='Open Image', command=open_image)
open_button.pack()

# Tạo button để nén ảnh
compress_button = tk.Button(root, text='Compress Image', command=compress_image)
compress_button.pack()

# Tạo figure và canvas để hiển thị ảnh
fig_original, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(fig_original, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Khởi tạo biến toàn cục cho ảnh grayscale và rank
img_gray = None
rank_to_compress = 10

# Hiển thị cửa sổ tkinter
root.mainloop()
