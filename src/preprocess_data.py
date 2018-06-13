#!/usr/bin/env python3
"""
    Pre-processing image data, resizing and saving to a new folder. Then all images are converted into numpy arrays with
        dimensions: (x_dim, y_dim, 3), the expected size for the the DCGAN.

    Code here heavily references Anuj Shah's code on loading datasets into CNNs, the source code can be found here:
    https://github.com/anujshah1003/own_data_cnn_implementation_keras/blob/master/custom_data_cnn.py
"""
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


def resize_images(style_name):
    """
    Helper function used to resize all images in the specified source directory to the specified dimensions.
    Requires path to data source, and path to output directory where the resized images is stored.
    Currently configured to resize images to 64x64x3.
    All output images are saved as JPEGs.
    :return: None
    """
    # disable decompression bomb warning, trusting that all images being loaded are safe
    Image.MAX_IMAGE_PIXELS = None
    # load data
    data_src = os.path.abspath('../data/{}'.format(style_name))
    os.mkdir('../flask_app/static/images/processed_{}'.format('_'.join(style_name.lower().split(' '))))
    output_dir = ('../flask_app/static/images/processed_{}'.format('_'.join(style_name.lower().split(' '))))
    # variables for resizing images
    img_rows = 64
    img_cols = 64

    data_list = os.listdir(data_src)
    # resizing images and saving it to specified output directory
    for file in data_list:
        indx = data_list.index(file)
        src = Image.open(os.path.join(data_src, file))
        if src.mode != 'RGB':
            # TODO: fix image corruption
            # some images have an alpha channel, we only need RGB for the network
            # removing the alpha channel could potential alter the image's color
            src = src.convert('RGB')
        try:
            img = src.resize((img_rows, img_cols))
        except Exception:
            pass
        img.save(os.path.join(output_dir, file))
        if indx % 100 == 0:
            print('Successfully resized {0}, {1} images remaining.'.format(indx, len(data_list) - indx))
    print('All images have been resized successfully!')


def load_data(data_dir):
    """
    Called by the DCGAN to load in the training dataset. Reads images in the dataset and converts them to numpy arrays.
    Requires path to the source, and the path to the preprocessed data.
    Currently configured for 64x64x3 images.
    :params:
    data_src = path to images directory
    :return:
    X_train = numpy array of all training images partitioned from the specified dataset
    X_test = numpy array of all test images partitioned from the specified dataset
    y_train = numpy array of labels for all training images
    y_test = numpy array of labels for all test images
    """
    # variables for resizing images
    img_rows = 64
    img_cols = 64

    imlist = os.listdir(data_dir)

    # converting each image into a np array, the arrays form a matrix
    im_arrays = []
    corrupted_imgs = 0
    for im in imlist:
        if im != '.DS_Store':
            im_array = np.array(Image.open(os.path.join(data_dir, im)))
            if im_array.shape == (img_rows, img_cols, 3):
                im_arrays.append(im_array.flatten())
            else:
                corrupted_imgs += 1
    im_matrix = np.array(im_arrays)
    num_samples = len(os.listdir(data_dir)) - corrupted_imgs
    # labeling the image arrays, necessary for training but we only have one label for the data set
    label = np.ones((num_samples,), dtype=int)
    label[0:num_samples] = 1

    train_data = [im_matrix, label]

    (X, y) = (train_data[0], train_data[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    return X_train, X_test, y_train, y_test


