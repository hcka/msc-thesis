# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:24:42 2018

@author: Lia Morra
"""
import os, shutil, cv2, csv, keras, re, argparse, urllib.request
import numpy as np
import hashlib as hs
import datetime as dt
from pathlib import Path
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

# -----------------------------------------------------------------------------
#%% Additional functions
# Check if the given input is a path
def osDirCheck(path):
    if (not os.path.isdir(path)): raise ValueError('{} Is not a valid path'.format(path))

# Small function to merge the datasets
def mergeDataSets(first_batch, second_batch):
    print('\nDatasets are merging...')
    no_folders = 0
    no_files = 0
    for folder in os.listdir(second_batch):
        source_folder = os.path.join(second_batch, folder)
        dest_folder = os.path.join(first_batch, folder)
        no_folders += 1
        if not os.path.exists(dest_folder): os.makedirs(dest_folder) 
        for file in os.listdir(os.path.join(second_batch, folder)):
            source_file = os.path.join(source_folder, file)
            shutil.copy(source_file, dest_folder)
            no_files += 1
    statsWrite(str(no_files)+' Files moved in '+str(no_folders)+' folders')
    print(no_files, 'Files moved in', no_folders,' folders')

# Write events to a csv file    
def statsWrite(text):
    if os.path.exists(stats_dir): mode = 'a'
    else: mode = 'w'
    split_text = text.split()
    with open(stats_dir, mode=mode, newline='') as stats:
        csv_writer = csv.writer(stats, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(split_text)
    stats.close()

# Check the available files the checksum file exists
def md5check(checks_csv):
    existed_files = []
    if os.path.exists(checks_csv):         
        with open(checks_csv, 'r', newline='') as discardedCsv:
            reader = csv.reader(discardedCsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                splitted = re.findall(r"[\w']+", row[1])
                existed_files.append(splitted[-2]+'.'+splitted[-1])
        discardedCsv.close()
    return existed_files
    
# -----------------------------------------------------------------------------
#%% Data cleaning functions
# First part to clean the non photo images in the input directory
# Check white pages
def checkWhitePages(img):
    perw = 1 - cv2.countNonZero(255 - img) /  (img.shape[0] * img.shape[1])
    return  (perw)

# Run the classifier to assure the correct images are discarded
# FIXME no need to load the image twice, we can just resize it and preprocess it (line 139)
# pay attention to open cv 
def binaryClassifier(img, model):
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(img, (224, 224))
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = preprocess_input(test_image)
    result = model.predict(test_image)
    return result[0][0] >= 0.99
   
# Clean directory where each subdirs contains images from one single claim
def cleanDirectory(inp_dir, out_dir, out_file, stats):
    print('\nDirectory cleaning...')
    discard_dir = os.path.join(out_file, 'Discarded_images.csv')
    if os.path.exists(discard_dir): discard_mode = 'a'
    else: discard_mode = 'w'
    md5checks_dir = os.path.join(out_file, 'md5checks.csv')
    existed_files = md5check(md5checks_dir)
    if len(existed_files): chck_mode = 'a'
    else: chck_mode = 'w'
    
    #FIXME: if the model file does not exist, download it (line 318)
    # Thresholds
    model = keras.models.load_model(classifier_model)
    thresh_dim = 10000
    thresh_size = 144
    thresh_aspect_ratio = 0.3
    # File counters of directory cleaning
    num_folders = 0
    num_files = 0
    discarded_from_file06 = 0 
    discarded_from_file11 = 0
    discarded_from_others = 0
    # File counters of dHash creator
    from_file06 = 0 
    from_file11 = 0
    from_others = 0
    # Start cleaning and write outputs to csv files
    with open(discard_dir, discard_mode, newline='') as discardedImages, open(md5checks_dir, mode=chck_mode, newline='') as checksums:         
        discarded_writer = csv.writer(discardedImages, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        checksum_writer = csv.writer(checksums, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fold_list = os.listdir(inp_dir)
        # sort through folder, use tqdm to keep track of status
        for folder in tqdm(fold_list, desc='is in progress'):
            # Generate subdirs path
            num_folders += 1
            inp_folder = os.path.join(inp_dir, folder)
            out_folder = os.path.join(out_dir, folder)
            for file in os.listdir(inp_folder):
                # check the dimension
                num_files += 1
                source = os.path.join(inp_folder, file)
                discard=False
                if (os.stat(source).st_size < thresh_dim): discard=True
                # Read image
                if (not discard):
                    image=cv2.imread(source)
                    if (image is None):
                        discard = True
                        print("Unable to read image " + file)
                    else: 
                        # check the aspect ratio and size
                        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        aspect_ratio = np.min(image_gray.shape) / np.max(image_gray.shape)
                        if (aspect_ratio < thresh_aspect_ratio): discard = True
                        if (np.min(image_gray.shape) < thresh_size or np.max(image_gray.shape) < thresh_size): 
                            discard = True
                        else:          
                            # check if mostly white
                            perw = checkWhitePages(image_gray)                        
                            if (perw > 0.995): discard = True                                                                                        
                if (not discard): discard = binaryClassifier(image, model)
                # discard images by moving them to another destination
                if (discard):
                    destination = os.path.join(out_folder, file)
                    if (not os.path.exists(out_folder)):  os.makedirs(out_folder)
                    shutil.move(source, destination)
                    discarded_writer.writerow([source])
                    splitted = file.split('_')
                    if splitted[2] == '06': discarded_from_file06 += 1                        
                    elif splitted[2] == '11': discarded_from_file11 += 1                       
                    else: discarded_from_others += 1       

                #FIXME if not discarded calculated md5 to avoid reading all the image twices (below code)
                # Generate dHash if the image is not discarded
                if (not discard) and (image not in existed_files):                 
                    imageHash = dhash(image_gray)
                    # Write md5 checksum and file name in a csv file       
                    checksum_writer.writerow([imageHash, os.path.join(inp_dir, os.path.join(folder, file))])
                    splitted = file.split('_')
                    if splitted[2] == '06': from_file06 += 1                        
                    elif splitted[2] == '11': from_file11 += 1                       
                    else: from_others += 1                                          
    discardedImages.close()
    checksums.close()
    # Write directory cleaning results to stats file
    statsWrite('Directory cleaner')
    statsWrite('Found '+str(num_files)+' files in '+str(num_folders)+' folders')
    statsWrite(str(discarded_from_file06)+' Images discarded from 06 files')
    statsWrite(str(discarded_from_file11)+' Images discarded from 11 files')
    statsWrite(str(discarded_from_others)+' Images discarded from other files')
    statsWrite(' ')
    print('Total number of discarded images: ', discarded_from_file06+discarded_from_file11+discarded_from_others)
    # Write dHash generator results to stats file
    statsWrite('Checksum generator')
    statsWrite(str(from_file06)+' Images from 06 files')
    statsWrite(str(from_file11)+' Images from 11 files')
    statsWrite(str(from_others)+' Images from other files')
    statsWrite(str(len(existed_files))+' Files were already available')
    statsWrite(' ')     
    print('Total number of generated dHashes: ',from_file06+from_file11+from_others)

# -----------------------------------------------------------------------------
#%% Second part to detect and discard the duplicate photos
# Differential hash generator
def dhash(image, hashSize=8):
    # Resize the input image (9x8 pixels)
    resized = cv2.resize(image, (hashSize+1, hashSize))
    # Compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:]>resized[:, :-1]
    # Convert the difference image to a hash
    return hs.md5(diff).hexdigest()

# Clean the exact copies available in the same folder
def cleanDuplicates(inp_dir, out_dir, out_file, stats):
    md5checks_dir = os.path.join(out_file, 'md5checks.csv')
    # calculate md5 checksums     
    # FIXME verify if file exists and in that case, only calculate the new files (line 154)
    # alternative, calculate md5 checksum only for the new dataset 
    print('\nDuplicate cleaning...')
    found_in_file06 = 0 
    found_in_file11 = 0
    found_in_others = 0
    discarded_from_file06 = 0 
    discarded_from_file11 = 0
    discarded_from_others = 0
    # Detect the exact copies which are in the same folder
    with open(md5checks_dir, 'r', newline='') as checksums:
        reader = csv.reader(checksums, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)   
        to_discard = dict()
        to_list = dict()
        for row in reader:
            # Check if the row contains any data or it is empty
            if any (row):
                # Get the complete filepath from the row and seperate the folder path
                folderPath = os.path.dirname(row[1])
                # Check if the checksum already exists or not
                if row[0] in to_discard.keys():
                    # Get the corresponding folder path for comparison
                    checkPath = os.path.dirname(to_discard[row[0]][0])
                    to_list[row[0]].append(row[1])
                    splitted = row[1].split('_')
                    if splitted[-2] == '06': found_in_file06 += 1                        
                    elif splitted[-2] == '11': found_in_file11 += 1                       
                    else: found_in_others += 1
                    # Append if the folder paths are exact pairs
                    if folderPath == checkPath:
                        to_discard[row[0]].append(row[1])
                else:
                    to_discard[row[0]] = [row[1]]
                    to_list[row[0]] = [row[1]]
    checksums.close()
    # Remove the exact copies and list them in a csv file
    discarded_pairs_dir = os.path.join(out_file, 'Discarded_copies.csv')
    if os.path.exists(discarded_pairs_dir): discard_mode = 'a'
    else: discard_mode = 'w'
    with open(discarded_pairs_dir, discard_mode, newline='') as discarded_pairs:
        discarded_pairs_writer = csv.writer(discarded_pairs, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for hsh in tqdm(to_discard.keys(), desc='is in progress'): 
            listf = to_discard[hsh]          
            if (len(listf) > 1):
                for l in range(1,len(listf)):
                    # Get the complete filename
                    fname = to_discard[hsh][l]
                    path, file = os.path.split(fname)
                    path, folder = os.path.split(path)
                    # Get onl the last two parts of the path (image file and folder)
                    fname = os.path.join(folder, file)
                    # Generate source and destination paths
                    source = os.path.join(inp_dir, fname)
                    destination = os.path.join(out_dir, fname)
                    # Create a folder in the same name with the source
                    discard_folder = os.path.join(out_dir, folder)
                    if (os.path.exists(source)):
                        if (not os.path.exists(discard_folder)): os.makedirs(discard_folder)                              
                        shutil.move(source, destination)
                        discarded_pairs_writer.writerow([to_discard[hsh][0], os.path.join(path, fname)])
                        splitted = file.split('_')
                        if splitted[2] == '06': discarded_from_file06 += 1                        
                        elif splitted[2] == '11': discarded_from_file11 += 1                       
                        else: discarded_from_others += 1
    discarded_pairs.close()
    # Create another csv file to detect the remaining copies
    exact_copies_dir = os.path.join(out_file, 'Remaining_copies.csv')
    if os.path.exists(exact_copies_dir): mode = 'a'
    else: mode = 'w'
    with open(exact_copies_dir, mode, newline='') as exact_copies:
        exact_copies_writer = csv.writer(exact_copies, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for hsh in to_list.keys():
           listf = to_list[hsh]
           if (len(listf) > 1):
               for l in range(1,len(listf)):
                   fname = to_list[hsh][l]  
                   exact_copies_writer.writerow([to_list[hsh][0], fname])
    exact_copies.close()
    # Compare the two csv files and update remaining copies
    with open(discarded_pairs_dir, 'r', newline='') as discardedCsv, open(exact_copies_dir, 'r', newline='') as exactCsv:
        discarded_lines = discardedCsv.readlines()
        exact_lines = exactCsv.readlines()
    discardedCsv.close(), exactCsv.close()
    with open(exact_copies_dir, 'w', newline='') as exact_copies:
        for line in exact_lines:
            if line not in discarded_lines:
                exact_copies.write(line) 
    exact_copies.close()
    # Write found and discarded duplicates to stats file
    statsWrite('Duplicate cleaner')
    statsWrite(str(found_in_file06)+' Found from 06 files '+str(discarded_from_file06)+' discarded')
    statsWrite(str(found_in_file11)+' Found from 11 files '+str(discarded_from_file11)+' discarded')
    statsWrite(str(found_in_others)+' Found from other files '+str(discarded_from_others)+' discarded')
    statsWrite(' ')
    print('Total number of duplicates: ', found_in_file06+found_in_file11+found_in_others)
    print('Total number of discarded duplicates: ', discarded_from_file06+discarded_from_file11+discarded_from_others)
    
# -----------------------------------------------------------------------------    
#%% Main function and input output paths    
if __name__ == '__main__':
#    #FIXME pass directories through command line interface (below code)
#    # Get current working directory and download the classifier model
#    working_dir = os.getcwd()
#    classifier_model = os.path.join(working_dir, 'dataClean_classifier.h5')
#    if not os.path.exists(classifier_model):
#        print('Classifier model is downloading...')
#        url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
#        urllib.request.urlretrieve(url, classifier_model)
#        print('Download completed')
#    
#    # Get directories
#    ap = argparse.ArgumentParser(description='Data Claeaning: Discards non photo and duplicate images')
#    ap.add_argument('-n', '--new', required=True, help='First batch images to be processed', type=str)    
#    ap.add_argument('-ot', '--out', required=True, help='Out directory to save the discarded images', type=str)
#    ap.add_argument('-s', '--save', required=True, help='Save to save output files', type=str)
#    ap.add_argument('-o', '--old', required=False, help='Second batch of images to be merged', type=str)
#    args = vars(ap.parse_args())
#    
#    # Validate directories
#    osDirCheck(args['new'])
#    main_inp_dir = Path(args['new'])
#    osDirCheck(args['out'])
#    main_out_dir = Path(args['out'])
#    osDirCheck(args['save'])
#    out_file_dir = Path(args['save'])
#    second_batch_dir = args['old']
#    if second_batch_dir:
#        osDirCheck(args['old'])
#        second_batch_dir = Path(args['old'])
#    
#    # Initiate the process
#    stats_dir = os.path.join(out_file_dir, 'Cleaning_pipeline_stats.csv')
#    statsWrite('Data cleaning pipeline')
#    statsWrite('Start time: '+str(dt.datetime.now()))
#    statsWrite(' ')
#    
#    # Clean and merge the datasets
#    cleanDirectory(main_inp_dir, main_out_dir, out_file_dir, stats_dir)
#    if second_batch_dir: mergeDataSets(main_inp_dir, second_batch_dir)      
#    cleanDuplicates(main_inp_dir, main_out_dir, out_file_dir, stats_dir)
#    
#    # Finalize the process
#    statsWrite('Stop time: '+str(dt.datetime.now()))
#    print('\nDone')
    
    #FIXME pass directories through command line interface (below code)
    start_time = dt.datetime.now()
    working_dir = os.getcwd()
    # Get input/output directories as user inputs
    main_inp_dir = input('Input directory: ')
    osDirCheck(main_inp_dir)
    main_inp_dir = Path(main_inp_dir)

    main_out_dir = input('Output directory: ')
    osDirCheck(main_out_dir)
    main_out_dir = Path(main_out_dir)

    out_file_dir = input('Save directory: ')
    osDirCheck(out_file_dir)
    out_file_dir = Path(out_file_dir)
    stats_dir = os.path.join(out_file_dir, 'Cleaning_pipeline_stats.csv')
    
    second_batch_dir = input('Dataset to be merged: ')
    if second_batch_dir:
        osDirCheck(second_batch_dir)
        second_batch_dir = Path(second_batch_dir)
    # Download the classifier model if it does not exist in the current directory
    # FIXME I do not know where to uplad the model. Below url contains a cat photo.
    classifier_model = os.path.join(working_dir, 'dataClean_classifier.h5')
    if not os.path.exists(classifier_model):
        print('Classifier model is downloading...')
        url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
        urllib.request.urlretrieve(url, classifier_model)
        print('Download completed')
         
    #FIXME merge datasets after cleaning the first directory (line 337)
    #if we were to add a new batch of images, we would already have the md5 checksum calculated
    # for all previous images, so this should be changed
    
    statsWrite('Data cleaning pipeline')
    statsWrite('Start time: '+str(dt.datetime.now()))
    statsWrite(' ')
    
    cleanDirectory(main_inp_dir, main_out_dir, out_file_dir, stats_dir)
    if second_batch_dir: mergeDataSets(main_inp_dir, second_batch_dir)      
    cleanDuplicates(main_inp_dir, main_out_dir, out_file_dir, stats_dir)
     
    statsWrite('Stop time: '+str(dt.datetime.now()))
    print('\nDone')