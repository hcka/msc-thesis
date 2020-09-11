# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:11:11 2018

@author: Admin
"""

import math, csv, os, itertools
import matplotlib.pyplot as plt
import numpy as np, pickle as pk, datetime as dt
from itertools import cycle
from scipy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import models, layers, optimizers
from utils import RotNet
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

#%% Additional functions
# Save numerical outputs to a pickle file
def pickleSave(dictionary):
    with open(os.path.join(save_dir, 'out.pickle'), 'ab') as file_pi:
        pk.dump(dictionary, file_pi)
    file_pi.close()

# Write incidents to a csv file
def statsWrite(text):
    stats_csv = os.path.join(save_dir, 'mnist_orienter_stats.csv')
    if os.path.exists(stats_csv): mode = 'a'
    else: mode = 'w'
    split_text = text.split()
    with open(stats_csv, mode=mode, newline='') as stats:
        csv_writer = csv.writer(stats, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(split_text)
    stats.close()

# Create the fine tuned model
def createModel():
    # Create a sequential model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return(model)

#%% Plot functions      
# History plot of the CNN model
def historyPlots(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    xint = range(min(epochs), math.ceil(max(epochs))+1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy');
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.xticks(xint); plt.legend(); fig1 = plt.gcf()
    plt.tight_layout(); plt.show()
    fig1.savefig(os.path.join(save_dir, 'train-val_acc.png'), dpi=100)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
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
        plt.text(j, i, format(cm[i, j], fmt),
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
    colors = cycle(['blue', 'darkorange', 'green'])
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

# Validatiion and test errors
def results(item, check):
    # Initialize variables
    fnames = y_val
    ground_truth = item.classes
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
    
    # Generate correct step number for each step
    if check == 'Validation': 
        steps = len(item.class_indices)*len(y_val)/item.samples
        filenames = fnames
        for i in range(len(item.class_indices)-2): fnames += filenames     
    else: 
        steps = item.samples/item.batch_size
    
    # Make predictions and find errors
    predictions = model.predict_generator(item, steps=steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    pred_confidence = np.amax(predictions, axis=1)
    errors = np.where(predicted_classes != ground_truth)[0]
    print("{} errors = {}/{}".format(check,len(errors),len(item.classes)))
    
    # List errors into the stats.csv
    statsWrite(' ')
    statsWrite(check+' errors: '+str(len(errors)))
    statsWrite('Actual-Pred-Confidence/Indice')
    statsWrite(idx_list)
    for i in range(len(errors)):          
        pred_result = [ground_truth[errors[i]], predicted_classes[errors[i]], 
                       pred_confidence[errors[i]], ',', errors[i]]
        pred_result = ' '.join(map(str, pred_result))
        statsWrite(pred_result)
    
    # Generate plots and return values
    values['ground_truth'] = ground_truth
    values['predictions'] = predictions
    values['predicted_classes'] = predicted_classes
    values['label2idx'] = label2idx
    
    rocCurve(ground_truth, predictions, label2idx, check)
    confMatrix(ground_truth, predicted_classes, label2idx, check, normalize=False)
    return values

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    root_dir = os.getcwd()
    save_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    statsWrite('MNIST')
    statsWrite('Start time: '+str(dt.datetime.now()))
    
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    
    x_train = x_train[:100]
    y_train = y_train[:100]
    
    x_val = x_val[:20]
    y_val = y_val[:20]
    
    x_tst = x_val[:-20]
    y_tst = y_val[:-20]
    
    nb_train_samples, img_rows, img_cols = x_train.shape
    nb_val_samples = x_val.shape[0]
        
    train_batchsize = 10
    val_batchsize = 5
    tst_batchsize = 5
    epochs = 10
    lr = 1e-4
    
    image_size = 224
    target_angles = [0, 180, 270, 90]
    
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze n number of layers from the last
    for layer in vgg_conv.layers[:]:
        layer.trainable = False
    
    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)
        
    # Create a sequential model
    model = createModel()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])
    
    # Callbacks    
    # Create a callbacks list
    checkpoint = ModelCheckpoint(os.path.join(save_dir, 'best_model.h5'), monitor='val_acc', 
                                 verbose=1, save_best_only=True, mode='max')    
    early_stopping = EarlyStopping(patience=4)
    tensorboard = TensorBoard()
    callbacks=[checkpoint, early_stopping, tensorboard]
    
    # Data Generator for train data        
    train_gen = RotNet(x_train, target_size=(image_size,image_size),target_angles=target_angles,
                       batch_size=train_batchsize, preprocess_func=preprocess_input, shuffle=True)

    # Data Generator for validation data
    val_gen = RotNet(x_val, target_size=(image_size,image_size),target_angles=target_angles,
                     batch_size=val_batchsize, preprocess_func=preprocess_input, shuffle=True)
    
    # Data Generator for test data
    tst_gen = RotNet(x_tst, target_size=(image_size,image_size),target_angles=target_angles,
                     batch_size=tst_batchsize, preprocess_func=preprocess_input, shuffle=True)
    
#    # Training loop
#    history = model.fit_generator(train_gen, steps_per_epoch=nb_train_samples/train_batchsize,
#                                  epochs=epochs, validation_data=val_gen, 
#                                  validation_steps=nb_val_samples/val_batchsize,
#                                  callbacks=callbacks, verbose=1) 
    
    # Save model and model history
    model.save(os.path.join(save_dir, 'model.h5'))
    model.save_weights(os.path.join(save_dir, 'model_wieghts.h5'))
    pickleSave(history.history)

    # Serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(save_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    json_file.close()    
    
    # Plot results
    val_outs = results(val_gen, 'Validation')
    pickleSave(val_outs)
    test_outs = results(tst_gen, 'Test')
    pickleSave(test_outs)
    historyPlots(history)
    
    # Write stop time to stats
    statsWrite('Stop time: '+str(dt.datetime.now()))
#
#    x,y = train_gen.next()
#    for i in range(0,4):
#        image = x[i]
#        plt.imshow(image)
#        plt.show()    