import numpy as np
import os

def npy_save_txt(name, data, save_to_temp=True):
    if save_to_temp:
        save_path = os.path.join('tmp', f"{name}.txt")
        dirname = os.path.dirname(save_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        np.savetxt(save_path, data)

if __name__ == "__main__":
    npy_save_txt("123/456", np.array([[1,2,3], [4,5,6]]))