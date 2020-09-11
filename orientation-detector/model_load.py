# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:38:05 2018

@author: Cagri
"""
"""
utils_orn:
Model loader and result re-plotter in case of necessity to re-produce the results.
Results can be plotted either using "out.pickle" file or running the predictor
for validation and test generators. If the results will be re-produced from the
predictions, batch sizes and other parameters must be checked before running the
code in order to assure the coherence of the re-produced results and the 
original ones.

Tis codefile must be in the same directory with "train", "validation" and "test"
folders and the application specific "RotNet" data generator. Output will be 
saved to "output" folder in the same directory.
"""
import keras, math, csv, os, itertools, cv2
import matplotlib.pyplot as plt
import numpy as np, pickle as pk, datetime as dt
from rotnet import RotNetGen
from itertools import cycle
from scipy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

#%% Additional functions-------------------------------------------------------
# Write events to a csv file
def statsWrite(text):
    stats_csv = os.path.join(save_dir, 'load_orienter_stats.csv')
    if os.path.exists(stats_csv): mode = 'a'
    else: mode = 'w'
    split_text = text.split()
    with open(stats_csv, mode=mode, newline='') as stats:
        csv_writer = csv.writer(stats, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(split_text)
    stats.close()

def pickleSave(dictionary):
    with open(os.path.join(save_dir, 'out.pickle'), 'ab') as file_pi:
        pk.dump(dictionary, file_pi)
    file_pi.close()    

#%% Plot functions-------------------------------------------------------------  
# History plot of the CNN model
def historyPlots(history, fsize=15):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))
    xint = range(min(epochs), math.ceil(max(epochs))+1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Train and validation accuracy', fontsize=fsize)
    plt.xlabel('Epochs', fontsize=fsize); plt.ylabel('Accuracy', fontsize=fsize);
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.xticks(xint); plt.legend(); fig = plt.gcf()
    plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, 'train-val_acc.png'), dpi=100)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Train and validation loss', fontsize=fsize)
    plt.legend();plt.xlabel('Epochs', fontsize=fsize); plt.ylabel('Loss', fontsize=fsize);
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.xticks(xint);plt.legend(); fig = plt.gcf()
    plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, 'train-val_loss.png'), dpi=100)

# Confusion matrix
def confMatrix(true_labels, pred_labels, classes, check, normalize=False, fsize=15):
    cmap=plt.cm.Blues
    cm = confusion_matrix(true_labels, pred_labels)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]     
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=fsize)
    plt.yticks(tick_marks, classes, rotation=0, fontsize=fsize)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=fsize,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")    
    plt.title(check+' confusion matrix', fontsize=fsize)    
    plt.xlabel('Predicted label', fontsize=fsize); plt.ylabel('True label', fontsize=fsize)    
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, check+'_conf_matrix.png'), dpi=100)
    
# ROC curves   
def rocCurve(true_labels, predictions, classes, check, fsize=15):       
    n_classes = len(classes)
    true_labels = label_binarize(true_labels, np.linspace(0,n_classes,n_classes+1))
    true_labels = true_labels[:,:-1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i]) 
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])      
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))   
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])  
    # Finally average it and compute AUC
    mean_tpr /= n_classes   
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])    
    # Plot all ROC curves
    plt.figure()
    if n_classes > 2:
        plt.plot(fpr["micro"], tpr["micro"], label='micro-avg. ROC curve (AUC = {0:0.2f})'
                 ''.format(roc_auc["micro"]), color='darkorange', linestyle=':', linewidth=3)   
        plt.plot(fpr["macro"], tpr["macro"], label='macro-avg. ROC curve (AUC = {0:0.2f})'           
                 ''.format(roc_auc["macro"]), color='indigo', linestyle=':', linewidth=3)            
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC curve of class {0} (AUC = {1:0.2f})' ''.format(i, roc_auc[i]))                
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fsize)
    plt.ylabel('True Positive Rate', fontsize=fsize)
    plt.title(check+' ROC curve', fontsize=fsize)
    plt.legend(loc="lower right")
    plt.grid()
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, check+'_ROC_curve.png'), dpi=100)


# Unidentified images prediction results
def unidentPredictions(item):
    fnames = item.filenames
    predictions = model.predict_generator(item, steps=item.samples/item.batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    pred_confidence = np.amax(predictions, axis=1)
    unin_predicted_classes = np.argmax(predictions, axis=1)
    unique, counts = np.unique(unin_predicted_classes, return_counts=True)
    pred_result = ' '.join(map(str, [unique, counts]))                              
    statsWrite(' ')
    statsWrite('Unidentified images predictions results: '+pred_result)
    statsWrite('Pred-Confidence/F.name')
    for i in range(len(fnames)):          
        pred_result = [predicted_classes[i], pred_confidence[i], ',', fnames[i]]                       
        pred_result = ' '.join(map(str, pred_result))
        statsWrite(pred_result)
    

# Validation and test errors
def results(item, check):
    # Initialize variables 
    fnames = item.filenames
    label2idx = item.class_indices
    idx2label = dict((v,k) for k,v in label2idx.items())
    values = {}
    idx_list = []
    for key, value in idx2label.items():
        key = str(key)+':'
        value = str(value)
        temp = [key, value]
        idx_list += temp
    idx_list = ' '.join(map(str, idx_list))
    
    # Generate correct step number for each call
    steps = item.samples/item.batch_size
    
    # Make predictions and find errors
    predictions = model.predict_generator(item, steps=steps, verbose=1)
    ground_truth = item.classes
    predicted_classes = np.argmax(predictions, axis=1)
    for i in range(len(predicted_classes)):
        if np.max(predictions[i])<0.9:
            predicted_classes[i] = 0
    pred_confidence = np.amax(predictions, axis=1)
    errors = np.where(predicted_classes != ground_truth)[0]
    print("{} errors = {}/{}".format(check,len(errors),len(item.classes)))
    
    # List errors into the stats.csv
    statsWrite(' ')
    statsWrite(check+' errors: '+str(len(errors)))
    statsWrite('Actual-Pred-Confidence/F.name')
    statsWrite(idx_list)
    for i in range(len(errors)):          
        pred_result = [ground_truth[errors[i]], predicted_classes[errors[i]], 
                       pred_confidence[errors[i]], ',', fnames[errors[i]]]
        pred_result = ' '.join(map(str, pred_result))
        statsWrite(pred_result)
    
    # Generate plots, return values and write values to the pickle file
    values['ground_truth'] = ground_truth
    values['predictions'] = predictions
    values['predicted_classes'] = predicted_classes
    values['label2idx'] = label2idx
    pickleSave(values)
    
    # Plot and return the outputs
    rocCurve(ground_truth, predictions, label2idx, check)
    confMatrix(ground_truth, predicted_classes, label2idx, check, normalize=False)
    return values

#%% Main function to re-produce the results from the predictions---------------
if __name__ == '__main__':
    # Image directories
    root_dir = os.getcwd()
    train_dir = os.path.join(root_dir, 'train')
    validation_dir = os.path.join(root_dir, 'validation')
    test_dir = os.path.join(root_dir, 'rma_sim')
    save_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
      
    # Initialize stats file
    statsWrite('LOAD MODEL: ORIENTATION CORRECTION CLASSIFIER')
    statsWrite('Start time: '+str(dt.datetime.now()))
    
    val_batchsize = 128
    tst_batchsize = 128
    image_size = 224
    lr = 0.5e-3 
    target_angles = [0, 90, 180, 270, 'undef']
    tst_classes = dict(zip([str(i) for i in (target_angles)], list(range(len(target_angles)))))
        
    model = keras.models.load_model(os.path.join(root_dir, 'best_model.h5'), compile=False)
    
    # Data Generator for validation data
    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_generator = test_datagen.flow_from_directory(test_dir, 
                                                      target_size=(image_size, image_size), classes=tst_classes,
                                                      batch_size=tst_batchsize, class_mode='categorical', shuffle=False)

    test_outs = results(test_generator, 'Test')
    
    statsWrite('Stop time: '+str(dt.datetime.now()))
    print('DONE SUCCESSFULLY')

#%% Results can be plotted from the pickle file    
    # Get numerical outputs from the pickle file and plot the results
    # 0: history, 1: validation, 2: test
    out_vals = []
    with (open(os.path.join(root_dir, 'out.pickle'), "rb")) as outs_pickle:
        while True:
            try:
                out_vals.append(pk.load(outs_pickle))
            except EOFError:
                break
    # History plots
    historyPlots(out_vals[0])
    
    # Validation plots
    rocCurve(out_vals[0]['ground_truth'], out_vals[0]['predictions'], 
             out_vals[0]['label2idx'], 'Holidays balanced')
    confMatrix(out_vals[0]['ground_truth'], out_vals[0]['predicted_classes'], 
               out_vals[0]['label2idx'], 'Holidays balanced', normalize=False)
    
    # Test plots
    rocCurve(out_vals[2]['ground_truth'], out_vals[2]['predictions'], 
             out_vals[2]['label2idx'], 'Test')
    confMatrix(out_vals[2]['ground_truth'], out_vals[2]['predicted_classes'], 
               out_vals[2]['label2idx'], 'Test', normalize=False)
#