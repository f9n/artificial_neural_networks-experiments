#!/usr/bin/env python3
import os
import skimage
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT = {
    'train': {
        'park': "trafikisaretleri/egitim/parketme-durma/",
        'danger': "trafikisaretleri/egitim/tehlike-uyari/",
    },
    'test': {
        'park': "trafikisaretleri/test/parketme-durma/",
        'danger': "trafikisaretleri/test/tehlike-uyari/",
    },
    'template_image': "templates/triangle.png",
    'MAX_EPOCH': 20
}

def get_all_file_list_in_folder_with_path(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

def get_image_pixels_with_scikit(image_path):
    img = skimage.io.imread(image_path)
    return img

def get_image_pixels_with_cv2(image_path):
    img = cv2.imread(image_path)
    return img

def is_pixel_blue(pixel):
    """ B G R"""
    B, G, R = pixel
    return B > (G + 100) and B > (R + 200)

def is_pixel_red(pixel):
    """ B G R"""
    B, G, R = pixel
    return R > (G + 100) and R > (B + 200)

def is_there_a_certain_color_in_image(pixels, callback):
    for pixel_row in pixels:
        for pixel_column in pixel_row: 
            color_status = callback(pixel_column)
            if color_status:
                return True
    return False

def is_there_a_triangle_in_image_with_template_method(img_rgb):
    # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
 
    # Read the template
    template = cv2.imread(PROJECT['template_image'],0)
 
    # Store width and heigth of template in w and h
    w, h = template.shape[::-1]
 
    # Perform match operations.
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
 
    # Specify a threshold
    threshold = 0.5
 
    # Store the coordinates of matched area in a numpy array
    loc = np.where( res >= threshold) 
    (a, b) = loc
    if len(a) > 0 and len(b) > 0:
        return True
    else:
        return False

def get_perceptron_inputs(image_paths):
    inputs = []
    for image_path in image_paths:
        img = get_image_pixels_with_cv2(image_path)
        blue_status = is_there_a_certain_color_in_image(img, is_pixel_blue)
        triangle_status = is_there_a_triangle_in_image_with_template_method(img)
        inputs.append([blue_status, triangle_status, 1])

    return inputs

def get_perceptron_weight(train_inputs, train_labels, epoch):
    w = np.zeros((3,))
    for i in range(0, epoch):
        #print("\tEpoch: {} Weight: {}".format(i, w))
        for row, label in zip(train_inputs, train_labels):
            target = row.dot(w)>=0
            error = label - target
            w += error * row

    return w

def get_perceptron_test_result(test_input, test_label, w):
    accuracy = 0
    for row, label in zip(test_input,test_label):
        # target: True or False
        # label: 1 or 0
        target = row.dot(w) >= 0
        if target == label:
            accuracy += 1

    return accuracy/len(test_label)

def get_inputs_and_labels_for_images(folder_path, result):
    image_paths = get_all_file_list_in_folder_with_path(folder_path)
    inputs = get_perceptron_inputs(image_paths)
    labels = np.full(len(inputs), result)
    return inputs, labels

def get_confission_matrix(test_inputs, test_labels, w):
    matrix = np.zeros(shape=(2, 2))
    for row, label in zip(test_inputs, test_labels):
        target = row.dot(w) >= 0
        if target:
            if label:
                matrix[1][1] += 1
            else:
                matrix[0][1] += 1
        else:
            if label:
                matrix[1][0] += 1
            else:
                matrix[0][0] += 1

    return matrix

def main():
    # Train Stuff
    print("Train Park")
    train_park_inputs, train_park_labels = get_inputs_and_labels_for_images(PROJECT['train']['park'], 0)
    print("Train Danger")
    train_danger_inputs, train_danger_labels = get_inputs_and_labels_for_images(PROJECT['train']['danger'], 1)

    train_inputs = np.concatenate([train_park_inputs,train_danger_inputs])
    train_labels = np.concatenate([train_park_labels,train_danger_labels])

    # Test stuff
    print("Test Park")
    test_park_inputs, test_park_labels = get_inputs_and_labels_for_images(PROJECT['test']['park'], 0)
    print("Test Danger")
    test_danger_inputs, test_danger_labels = get_inputs_and_labels_for_images(PROJECT['test']['danger'], 1)

    test_inputs = np.concatenate([test_park_inputs,test_danger_inputs])
    test_labels = np.concatenate([test_park_labels,test_danger_labels])

    results = []
    epochs = range(0, PROJECT['MAX_EPOCH'])
    for i in epochs:
        w = get_perceptron_weight(train_inputs, train_labels, i)
        result = get_perceptron_test_result(test_inputs, test_labels, w)
        print("MAX_EPOCH: {}, Weight: {}, Result: {}".format(i, w, result))
        results.append(result)

    w = get_perceptron_weight(train_inputs, train_labels, PROJECT['MAX_EPOCH'])
    confission_matrix = get_confission_matrix(test_inputs, test_labels, w)
    print("Confission Matrix")
    print(confission_matrix)
    
    results[0] = 0
    print(results)
    plt.xlabel('Epoch')
    plt.ylabel('Accurancy')
    plt.plot(epochs, results)
    plt.show()
    
if __name__ == "__main__":
    main()

