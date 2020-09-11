# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:21:12 2018

@author: Admin
"""
import keras, math, csv, os, itertools, cv2
import matplotlib.pyplot as plt
import numpy as np, pickle as pk, datetime as dt
from itertools import cycle
from scipy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

#%% Additional functions--------------------------------------------------------
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
    
#%% Plot functions--------------------------------------------------------------   
# History plot of the CNN model
def historyPlots(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))
    xint = range(min(epochs), math.ceil(max(epochs))+1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Train and validation accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy');
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.xticks(xint); plt.legend(); fig1 = plt.gcf()
    plt.tight_layout(); plt.show()
    fig1.savefig(os.path.join(save_dir, 'train-val_acc.png'), dpi=100)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Train and validation loss')
    plt.legend();plt.xlabel('Epochs'); plt.ylabel('Loss');
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.xticks(xint);plt.legend(); fig2 = plt.gcf()
    plt.tight_layout(); plt.show()
    fig2.savefig(os.path.join(save_dir, 'train-val_loss.png'), dpi=100)
    

# Confusion matrix
def confMatrix(true_labels, pred_labels, classes, check, normalize=False):
    cmap=plt.cm.Blues
    cm = confusion_matrix(true_labels, pred_labels)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]     
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=0)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")    
    plt.title(check+' confusion matrix')    
    plt.xlabel('Predicted label'); plt.ylabel('True label')    
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, check+'_conf_matrix.png'), dpi=100)
    
    
# ROC curves   
def rocCurve(true_labels, predictions, classes, check):       
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
                 ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=3)   
        plt.plot(fpr["macro"], tpr["macro"], label='macro-avg. ROC curve (AUC = {0:0.2f})'           
                 ''.format(roc_auc["macro"]), color='aqua', linestyle=':', linewidth=3)            
    colors = cycle(['blue', 'darkorange', 'green', 'indigo'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC curve of class {0} (AUC = {1:0.2f})' ''.format(i, roc_auc[i]))                
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(check+' ROC curve')
    plt.legend(loc="lower right")
    plt.grid()
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, check+'_ROC_curve.png'), dpi=100)
    

# Validation and test errors
def results(generator, check, save_flag=True):
    # Initialize variables
    print(check, 'results')
    fnames = generator.filenames
    ground_truth = generator.classes
    label2idx = generator.class_indices
    idx2label = dict((v,k) for k,v in label2idx.items())
    idx_list=[]
    values = {}
    for key, value in idx2label.items():
        key = str(key)+':'
        value = str(value)
        temp = [key, value]
        idx_list += temp
    idx_list = ' '.join(map(str, idx_list))
        
    # Make predictions and find errors
    predictions = model.predict_generator(generator, steps=generator.samples/generator.batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    pred_confidence = np.amax(predictions, axis=1)
    errors = np.where(predicted_classes != ground_truth)[0]
    print("{} errors = {}/{}".format(check,len(errors),len(generator.classes)))
    
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
    if save_flag: pickleSave(values)
    
    rocCurve(ground_truth, predictions, label2idx, check)
    confMatrix(ground_truth, predicted_classes, label2idx, check, normalize=False)
    return values

#%% Main function to load the model file and make new predictions---------------
if __name__ == '__main__':
    # Image directories
    root_dir = os.getcwd()
    validation_dir = os.path.join(root_dir, 'validation')
    test_dir = os.path.join(root_dir, 'test')
    save_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    # Initialize stats file   
    statsWrite('LOAD MODEL: DATA CLEAN CLASSIFIER')
    statsWrite('Start time: '+str(dt.datetime.now()))
    
    # Initialize the parameters    
    image_size = 224
    val_batchsize = 100
    tst_batchsize = 100
    lr = 1e-4
    
    # load model
    model = keras.models.load_model(os.path.join(root_dir, 'best_model.h5'))
    
    # Data generator for Validation data
    validation_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(image_size, image_size),
                                                                  batch_size=val_batchsize, 
                                                                  class_mode='categorical', shuffle=False)
    
    # Data generator for Test data
    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_size, image_size),
                                                      batch_size=tst_batchsize, 
                                                      class_mode='categorical', shuffle=False)
    
    # Write file numbers to stats
    statsWrite(' ')
    statsWrite('Validation files: '+str(validation_generator.samples))
    statsWrite('Test files: '+str(test_generator.samples))
    
    save_results = True
    val_outs = results(validation_generator, 'Validation', save_results)
    test_outs = results(test_generator, 'Test', save_results)
    
    statsWrite('Stop time: '+str(dt.datetime.now()))       
                   
    """
    # Import output history
    history = []
    with (open(history_file, "rb")) as openfile:
        while True:
            try:
                history.append(pickle.load(openfile))
            except EOFError:
                break
    historyPlots(history)
    """