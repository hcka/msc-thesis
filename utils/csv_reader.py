# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 01:38:58 2018

@author: Admin
"""
#%% Orientation classes
import os, csv

inpath = 'D:\\Lectures\\Thesis\\original_dataset\\second_batch'
csv_path = 'D:\Lectures\Thesis\original_dataset\scndbatch_orientations.csv'

with open(csv_path, 'w', newline='') as discardedCsv:
    writer = csv.writer(discardedCsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for subdir, dirs, image_files in os.walk(inpath):
        for file in image_files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_name = (os.path.join(subdir, file))
                splitted = file_name.split('\\')  
                writer.writerow([splitted[-2], splitted[-1]])  

#%% Error detection
import csv, os, shutil

csv_path = 'D:\\Lectures\\Thesis\\orienter\\errs.csv'
src_root = 'D:\\Lectures\\Thesis\\orienter\\dataset\\test'
dst_root = 'D:\\Lectures\\Thesis\\orienter\\dataset\\tst_errs'

err_files=[]
with open(csv_path, 'r', newline='') as discardedCsv:
    reader = csv.reader(discardedCsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        err_files.append(row)
discardedCsv.close()

for file in err_files:
    file_name = file[4]
    splitted = file_name.split('/')    
    src = os.path.join(src_root, file_name)
    dst = os.path.join(dst_root, splitted[0])
    shutil.copy(src, dst)

#%% Orientation classes
import os, csv, re

inpath = 'D:\\Lectures\\Thesis\orienter\\dataset\\test'
csv_path = 'D:\Lectures\Thesis\SampleData\\results\md5checks.csv'

files = []
with open(csv_path, 'r', newline='') as discardedCsv:
    reader = csv.reader(discardedCsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        splitted = re.findall(r"[\w']+", row[1])
        files.append(splitted[-2])
discardedCsv.close()

#%% Orientation error read
import os, csv, shutil

csv_path = 'D:\\Lectures\\Thesis\\orienter\\interim_out2\\val_preds.csv'

files = []
rows=[]
with open(csv_path, 'r', newline='') as discardedCsv:
    reader = csv.reader(discardedCsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        rows.append(row)
        splitted = row[4].split('/')
        file = os.path.join(splitted[2],splitted[3],splitted[4])
        files.append([row[0], row[1], splitted[4], file])
discardedCsv.close()


inpath = "D:\\Lectures\\Thesis\\orienter\\dataset"
opath="D:\\Lectures\\Thesis\\orienter\\interim_out2\\valerrs"

for row in files:
    src=os.path.join(inpath, row[3])
    splitted=row[3].split('\\')
    targ=row[0]+'_'+row[1]+'_'+row[2]
    targ=os.path.join(opath, splitted[1], targ)
    shutil.copy(src, targ)
