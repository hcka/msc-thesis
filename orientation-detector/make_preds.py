# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:25:48 2019

@author: Cagri
"""

import cv2, csv, os, shutil, keras
import numpy as np
import datetime as dt 
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def csvWrite(csvfname, text):
    if os.path.exists(csvfname): mode = 'a'
    else: mode = 'w'
    with open(csvfname, mode=mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(text)
    csvfile.close()
    
def rotate(name, image, angle, w_dir):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    
    # Get the backup before rotating the image
    shutil.copy(os.path.join(w_dir, name), os.path.join(w_dir, name+'.bak'))
    
    
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]])
    
    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    save_dir = os.path.join(w_dir, name)
    rotated = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    cv2.imwrite(save_dir, cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    
    
if __name__ == '__main__':
    print("Orientation experiment started at: ", dt.datetime.now())
    wdir = os.getcwd()
    images_path = "/scratch/work/progetti/RMA_ImageRecon/Data_Oct2018/ImageSet_v2.0_Dec2018_Grouped_Flat/Subset1"
    subs2out = "/scratch/work/progetti/RMA_ImageRecon/Data_Oct2018/ImageSet_v2.0_Dec2018_Grouped_Flat/subs1out"
    images_path = "D:\\Lectures\\Thesis\\orienter\\newpreds"
    subs2out = "D:\\Lectures\\Thesis\\orienter"
    if not os.path.exists(subs2out): os.mkdir(subs2out)
    images_path = os.path.join(wdir, 'newpreds')
    model_path = os.path.join(wdir, 'orn_classifier.h5')
    model = keras.models.load_model(model_path, compile=False)
    pred_results = os.path.join(subs2out, 'pred_results.csv')
    nonrotcsv = os.path.join(subs2out, 'nonrot.csv')
    lessthanth = os.path.join(subs2out, 'under_thresh.csv')
    statscsv = os.path.join(subs2out, 'rotation_stats.csv')
    rotate_dict = {1:270, 2:180, 3:90}
    detected_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
    rotated_dict = {1:0, 2:0, 3:0}
    underthresh_dict = {1:0, 2:0, 3:0}
    nonrotated = 0
    unreadible = 0
    frequencies = np.array([0.78, 0.065, 0.002, 0.058, 0.095])
    
    flist = os.listdir(images_path)
    splited_dir = images_path.split(os.path.sep)
    print("Found {} files in [{}] folder".format(len(flist), splited_dir[-1]))
    csvWrite(statscsv, ['Orientation new predictions '+splited_dir[-1]])
    csvWrite(statscsv, ['Started: ' + str(dt.datetime.now())])
    csvWrite(statscsv, ['Found '+str(len(flist))+' in '+splited_dir[-1]+' folder'])
    
    for fname in tqdm(flist):
        image = cv2.imread(os.path.join(images_path, fname))
        if not image is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test = cv2.resize(image, (224, 224))
            test = np.expand_dims(test, axis = 0)
            test = preprocess_input(test.astype('float64'))
            
            result = model.predict(test)
            weighted = np.multiply(result[0], frequencies)
            weighted = weighted.astype('float32')
            weighted = weighted/sum(weighted)
            
            pred_cls = weighted.argmax()
            detected_dict[pred_cls] += 1    
            
            if pred_cls != 0 and pred_cls != 4:
                rotate(fname, image, rotate_dict[pred_cls], images_path)                
                rotated_dict[pred_cls] += 1
            else: nonrotated += 1           
            csvWrite(pred_results, [pred_cls, weighted.max(), result[0], weighted, os.path.join(splited_dir[-1], fname)])
                
        else:
            unreadible += 1
            print(fname, " is unreadible")
                
    detected_dict = ' '.join("{!s}={!r}".format(key,val) for (key,val) in detected_dict.items())
        
    csvWrite(statscsv, ['Classes:',detected_dict])       
    csvWrite(statscsv, ['Rotated: ', rotated_dict])
    csvWrite(statscsv, ['Non rotated: ' + str(nonrotated)])
    csvWrite(statscsv, ['Finished: ' + str(dt.datetime.now())])
    print("Orientation experiment finished at: ", dt.datetime.now())
    
