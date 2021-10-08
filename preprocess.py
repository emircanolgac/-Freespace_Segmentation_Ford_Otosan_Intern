import glob
import cv2
import torch
import numpy as np
from constant import *


def tensorize_image(image_path_list, output_shape, cuda=False):

    local_image_list = []

    for image_path in image_path_list:

        image = cv2.imread(image_path)

        image = cv2.resize(image, output_shape)

        torchlike_image = torchlike_data(image)

        local_image_list.append(torchlike_image)

    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    if cuda:
        torch_image = torch_image.cuda()

    return torch_image


def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):

    local_mask_list = []

    for mask_path in mask_path_list:

        mask = cv2.imread(mask_path, 0)

        mask = cv2.resize(mask, output_shape)

        mask = one_hot_encoder(mask, n_class)

        torchlike_mask = torchlike_data(mask)


        local_mask_list.append(torchlike_mask)

    mask_array = np.array(local_mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

def image_mask_check(image_path_list, mask_path_list):

    if len(image_path_list) != len(mask_path_list):
        print("There are missing files ! Images and masks folder should have same number of files.")
        return False

    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('\\')[-1].split('.')[0]
        mask_name  = mask_path.split('\\')[-1].split('.')[0]
        if image_name != mask_name:
            print("Image and mask name does not match {} - {}".format(image_name, mask_name)+"\nImages and masks folder should have same file names." )
            return False

    return True


############################ TODO ################################
def torchlike_data(data):

    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))
    for ch in range(n_channels):
        torchlike_data[ch] = data[:,:,ch] 
        
    return torchlike_data

def one_hot_encoder(data, n_class):

    if len(data.shape) != 2:
        print("It should be same with the layer dimension, in this case it is 2")
        return
    if len(np.unique(data)) != n_class:
        print("The number of unique values ​​in 'data' must be equal to the n_class")
        return

    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int)

    for i, unique_value in enumerate(np.unique(data)):
        encoded_data[:,:,i][data==unique_value]=1

    return encoded_data
############################ TODO END ################################



if __name__ == '__main__':

    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()


    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()


    if image_mask_check(image_list, mask_list):

        batch_image_list = image_list[:BACTH_SIZE]

        batch_image_tensor = tensorize_image(batch_image_list, (224, 224))

        # Check
        print("For features:\ndtype is "+str(batch_image_tensor.dtype))
        print("Type is "+str(type(batch_image_tensor)))
        print("The size should be ["+str(BACTH_SIZE)+", 3, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_image_tensor.shape)+"\n")

        batch_mask_list = mask_list[:BACTH_SIZE]

        batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT, WIDTH), 2)

        # Check
        print("For labels:\ndtype is "+str(batch_mask_tensor.dtype))
        print("Type is "+str(type(batch_mask_tensor)))
        print("The size should be ["+str(BACTH_SIZE)+", 2, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_mask_tensor.shape))
        