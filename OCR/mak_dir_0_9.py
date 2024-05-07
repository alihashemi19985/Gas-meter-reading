import os
save_dir = r'C:\Users\Ali\Desktop\New folder'
for i in range(10):
    path_dir = os.path.join(save_dir, f'{i}')
    os.makedirs(path_dir)