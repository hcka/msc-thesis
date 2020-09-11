# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:17:55 2018

@author: Cagri

VGG16 based fine tuned model. The model classifies input images as relevant or
irrelevant. Binary classifier. Requiers Tensorflow, Keras and other dependencies
to generate the performance metrics plots. Generates a stats file in csv format
and saves all the plots to the given output directory.

"""

import numpy as np, seaborn as sns, pickle as pk
import math, csv, os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import models, layers, optimizers
#%% Function to build the model on preprocessed VGG16 net
def createModel():
    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return(model)
#%% Functions to visualize the results  
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
    
def confMatrix(actualLabels, predLabels, check):
    cm = confusion_matrix(actualLabels, predLabels)
    sns.heatmap(cm, annot=True, cmap="Set2")
    fname = check+'_conf_matrix.png'
    plt.title(check+' confusion matrix')    
    plt.xlabel('Predicted label'); plt.ylabel('True label')    
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(os.path.join(save_dir, fname), dpi=100)

def results(item, check):
    fnames = item.filenames
    ground_truth = item.classes
    label2index = item.class_indices
    idx2label = dict((v,k) for k,v in label2index.items())
    predictions = model.predict_generator(item, steps=item.samples/item.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),item.samples))
    # Show the errors
    if check == 'Validation': inp_path = validation_dir      
    elif check == 'Test': inp_path = test_dir
    roc_path = os.path.join(save_dir, check+'_roc.png')
    wrong_preds_path = check+'_errors.csv'
    with open(os.path.join(save_dir, wrong_preds_path), mode='w', newline='') as wrong_preds:
        csv_writer = csv.writer(wrong_preds, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(errors)):
            pred_class = np.argmax(predictions[errors[i]])
            pred_label = idx2label[pred_class]    
            title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(fnames[errors[i]].split('/')[0], pred_label, predictions[errors[i]][pred_class])
            original = load_img('{}/{}'.format(inp_path,fnames[errors[i]]))
            plt.figure(figsize=[7,7])
            plt.axis('off')
            plt.title(title)
            plt.imshow(original)
            plt.show()
            category, file = fnames[errors[i]].split('\\')
            csv_writer.writerow([file, category, predictions[errors[i]][pred_class]]) 
    wrong_preds.close()
    actual = np.concatenate((np.ones(100), np.zeros(100)), axis=0)
    fpr, tpr, thresh = roc_curve(actual, predictions[:,0])
    area = auc(fpr, tpr)       
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(area))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(check+' ROC curve')
    plt.legend(loc='best')
    plt.grid()
    fig = plt.gcf(); plt.tight_layout(); plt.show()
    fig.savefig(roc_path, dpi=100)        
    confMatrix(ground_truth, predicted_classes, check)
#%% Main function
if __name__ == '__main__':
    start_time = dt.datetime.now()
    # Image directories
    train_dir = 'D:/Lectures/Thesis/cleaner/set2/train'
    validation_dir = 'D:/Lectures/Thesis/cleaner/set2/validation'
    test_dir = 'D:/Lectures/Thesis/cleaner/set2/test'
    save_dir = 'D:/Lectures/Thesis/cleaner/set2'
    #    train_dir = '/ubuntu/anaconda3/cleaner/set2/train'
    #    validation_dir = '/ubuntu/anaconda3/cleaner/set2/validation'
    #    test_dir = '/ubuntu/anaconda3/cleaner/set2/test'
    #    save_dir = '/ubuntu/anaconda3/cleaner'
    # Initial parameters
    image_size = 224
    train_batchsize = 32
    val_batchsize = 10
    tst_batchsize = 10
    epochs = 25
    lr = 1e-4
    
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # Freeze n number of layers from the last
    for layer in vgg_conv.layers[:]:
        layer.trainable = False
    
    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)
    
    # Create the model
    model = createModel()
    # Preprocess the images 
    #train_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, zoom_range=0.2,
                                       rotation_range=20, horizontal_flip=True)
    #validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    #test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    # Data Generator for Training data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), 
                                                        batch_size=train_batchsize, class_mode='categorical')
    # Data Generator for Validation data
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(image_size, image_size),
                                                                  batch_size=val_batchsize, class_mode='categorical', shuffle=False)
    # Data Generator for Test data
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_size, image_size),
                                                                  batch_size=tst_batchsize, class_mode='categorical', shuffle=False)
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])
    # Create checkpoint
    checkpoint = ModelCheckpoint(os.path.join(save_dir, 'best_model.h5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(patience=2)
    tensorboard = TensorBoard()
    callbacks_list = [checkpoint, early_stopping, tensorboard]
    # Train the Model
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples/train_generator.batch_size, epochs=epochs,
                                  validation_data=validation_generator, validation_steps=validation_generator.samples/validation_generator.batch_size, 
                                  callbacks=callbacks_list, verbose=1)
    # Serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(save_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save(os.path.join(save_dir, 'model.h5'))
    model.save_weights(os.path.join(save_dir, 'model_wieghts.h5'))
    # Save model history
    with open(os.path.join(save_dir, 'trainHistoryDict.pickle'), 'wb') as file_pi:
        pk.dump(history.history, file_pi)
    file_pi.close()
    # Plot the accuracy and loss curves
    results(validation_generator, 'Validation')
    results(test_generator, 'Test')
    historyPlots(history)
    print('Start time: ', start_time, '\nStop_time: ', dt.datetime.now())
