# -*- coding: utf-8 -*-

# !pip install labelme
# !pip install tensorflow
# !pip install opencv-python
# !pip install albumentations
# !pip install matplotlib
# !pip uninstall -y opencv-python-headless
# !pip uninstall -y opencv-python
# !pip install opencv-python
# !pip install scikit-learn
# !pip install numpy
# !pip install seaborn


import os
import uuid
import time
import cv2

imPath = os.path.join('data', 'images')
imageNum = 20

# Use webcam
Capture = cv2.VideoCapture(0)

# Check if webcam opened
if not Capture.isOpened():
    print("Open Error")
    exit()

# Take and store images
for image in range(imageNum):
    # Capture frame
    ret, frame = Capture.read()

    # Check if frame can be read
    if not ret:
        print("Read Error")
        break

    nameImage = os.path.join(imPath, f'{str(uuid.uuid1())}.jpg') # Random image name
    cv2.imwrite(nameImage, frame) # Save image
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Capture.release()
cv2.destroyAllWindows()

# Open LabelMe to annoate images
!labelme

"""2. Load Images into Pipeline"""

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

images = None

images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False) # Load images into TensorFlow Data Pipeline

# Read in image and output pixel values
def load_image(path):
    encode = tf.io.read_file(path)
    pixels = tf.io.decode_jpeg(encode) # Decode each image file into a TensorFlow-compatible format
    return pixels

images = images.map(load_image) # Run function for all images in pipeline

"""3. Partition Images/Data"""

# Partition Images Manually: 70% training, 15% testing, 15% validation

# Map the images and labels (based on name) and add label file to the appropriate parition folder
for partition in ['train','test','val']:
    for file in os.listdir(os.path.join('data', partition, 'images')):
        img_name = file.split('.')[0]+'.json'
        img_path = os.path.join('data','labels', img_name)
        if os.path.exists(img_path):
            new_path = os.path.join('data', partition, 'labels', img_name)
            os.replace(img_path, new_path)

"""4. Test Augmentation"""

import albumentations as alb

# Apply transformations to images
augmentor = alb.Compose([alb.RandomCrop(width=400, height=400),
                         alb.HorizontalFlip(p=0.5),
                         alb.VerticalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.3),
                         alb.RandomGamma(p=0.3),
                         alb.RGBShift(p=0.3)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                  label_fields=['classes']))

# Load test image (test augmentation results for a sample image)
img = cv2.imread(os.path.join('data','train', 'images','0415f401-3086-4741-b960-19aee5b3637e.jpg'))

# Load test annotation
with open(os.path.join('data', 'train', 'labels', '0415f401-3086-4741-b960-19aee5b3637e.json'), 'r') as f:
    label = json.load(f)

# Store bboxes and class label for image
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640,480,640,480]))

coords

classes =['Raj']

# Apply augmentation
augmented = augmentor(image=img, bboxes=[coords], classes=classes) # class label -----------------------------------------------

# Draw bounding box around augmented image
cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [400,400]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [400,400]).astype(int)),
                    (255,0,0), 2)

plt.imshow(augmented['image'])

"""5. Use Augmentation on Entire Dataset"""

# Here we are iterating through dataset partitions
for dataa_split in ['train','test','val']:
    # We load the image file here
    for image in os.listdir(os.path.join('data', dataa_split, 'images')):
        imagePath = cv2.imread(os.path.join('data', dataa_split, 'images', image))
        coords = [0,0,0.00001,0.00001] # We initialize default bounding box values
        pathFileLabel = os.path.join('data', dataa_split, 'labels', f'{image.split(".")[0]}.json')

        #if the lable file exits we read and normalize corrdinates of the bounding box
        if os.path.exists(pathFileLabel):
            with open(pathFileLabel, 'r') as f:
                label = json.load(f)

            # we normalize bounding box corrdinates
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))

        # we will augment 10 imqges for each original image in the dataset so it will be about 1000 per person
        try:
            for x in range(10):
                # we apply augmentation transformation
                augmented = augmentor(image=imagePath, bboxes=[coords], classes=['Raj'])
                # then we save the augmented image with a new name
                cv2.imwrite(os.path.join('aug_data', dataa_split, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
                annotation = {}
                annotation['image'] = image

                # we first check if pathFileLabel exists
                if os.path.exists(pathFileLabel):
                    # if no bounding box exists for augmented image
                    if len(augmented['bboxes']) == 0:
                        # empty bounding box
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        # augmented bounding box
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    # if the file does not exist we will simply assign default values
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0

                # save thee annotations as json with the corresponding name
                with open(os.path.join('aug_data', dataa_split, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)

train_images = None
test_images = None
val_images = None

# Load augmented images to the TensorFlow data pipeline
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)
test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)
val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)

"""6. Prepare Labels"""

# Extract class label and bounding box coordinates from labels
def load_labels(path):
    with open(path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

train_labels = None
test_labels = None
val_labels = None

# Store labels in TensorFlow data pipeline
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

"""7. Combine Label and Image Samples"""

# Check partition lengths to make sure there are no errors
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

train_set = None
test_set = None
val_set = None

# Combine images and labels into one (final dataset)
# Apply shuffle, batch, and prefetch to optimize pipeline
train_set = tf.data.Dataset.zip((train_images, train_labels))
train_set = train.shuffle(3000)
train_set = train.prefetch(4)
train_set = train.batch(8)
test_set = tf.data.Dataset.zip((test_images, test_labels))
test_set = test.shuffle(750)
test_set = test.prefetch(4)
test_set = test.batch(8)
val_set = tf.data.Dataset.zip((val_images, val_labels))
val_set = val.shuffle(750)
val_set = val.prefetch(4)
val_set = val.batch(8)

train_set.as_numpy_iterator().next()[1]

train_samples = train_set.as_numpy_iterator()

next_sample = train_samples.next()

# View sample annotations from training set by draw boundin box
fig, axes = plt.subplots(ncols=4, figsize=(20,20))
for i in range(4):
    img = next_sample[0][i]
    bbox = next_sample[1][1][i]
    img_copy = img.copy()

    cv2.rectangle(img_copy,
                  tuple(np.multiply(bbox[:2], [120,120]).astype(int)),
                  tuple(np.multiply(bbox[2:], [120,120]).astype(int)),
                        (255,0,0), 2)

    axes[i].imshow(img_copy)

"""8. Build Model"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# VGG pretrained model
vgg = VGG16(include_top=False)

vgg.summary() # See layers in VGG model

# Leverage VGG model to build custom model
def build_model():
    input_layer = Input(shape=(120,120,3)) # Declare input (120x120 image)
    vgg = VGG16(include_top=False)(input_layer) # Pass input layer into VGG

    # Classification Model Layers
    class_pool = GlobalMaxPooling2D()(vgg)
    class_in = Dense(2048, activation='relu')(class_pool)
    class_out = Dense(1, activation='sigmoid')(class_in) # 1 output for classificaiton

    # Bounding Box Model Layers
    bbox_pool = GlobalMaxPooling2D()(vgg)
    bbox_in = Dense(2048, activation='relu')(bbox_pool)
    bbox_out = Dense(4, activation='sigmoid')(bbox_in) # 4 outputs for regression

    facetracker = Model(inputs=input_layer, outputs=[class_out, bbox_out]) # Classification and regression output
    return facetracker

facetracker = build_model()

facetracker.summary() # Layers for the customized model

X, y = train_set.as_numpy_iterator().next() # X=image, y=label

classes, coords = facetracker.predict(X) # Test initial model (before training)

classes, coords

"""9. Define Losses and Optimizers"""

# Calculate learning rate deacy
batches_per_epoch = len(train_set) # len(train) = 1554 (70% of dataset)
lr_decay = (1./0.75 -1)/batches_per_epoch

optimzer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay) # Configure optimizer

# Function to calculate localization error
def localization_loss(actual, pred):
    # Find squared difference of predicted and actual top left coordinate of bounding box
    diff_coord = tf.reduce_sum(tf.square(actual[:,:2] - pred[:,:2]))

    # Find heights and widths of bounding box
    actual_height = actual[:,3] - actual[:,1]
    actual_width = actual[:,2] - actual[:,0]
    pred_height = pred[:,3] - pred[:,1]
    pred_width = pred[:,2] - pred[:,0]

    # Find squared difference of predicted and actual height and width
    diff_hw = tf.reduce_sum(tf.square(actual_width - pred_width) + tf.square(actual_height-pred_height))

    return diff_coord + diff_hw

# Store errors/losses
classificaiton_loss = tf.keras.losses.BinaryCrossentropy()
regression_loss = localization_loss

"""10. Train Neural Network"""

# Here we handle training for the face tracker model
class FaceTracker(Model):
    def __init__(self, facetracker,  **kwargs):
        super().__init__(**kwargs)
        #initialize model facetracker for classification and localization
        self.model = facetracker

    # loss and optimizers
    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    # train neural network
    def train_step(self, batch, **kwargs):

        X, y = batch
        y[0].set_shape([None,1])

        # Track the gradients using TensorFlow's GradientTape
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            # Compute the classification and localization losses
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss+0.5*batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}

    # test model
    def test_step(self, batch, **kwargs):
        X, y = batch
        y[0].set_shape([None,1])

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss

        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

model = FaceTracker(facetracker) # Pass through neural network

model.compile(opt, classificaiton_loss, regression_loss) # Compile model using optimizera and losses

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=15, validation_data=val, callbacks=[tensorboard_callback]) # Train model

hist.history

# Plot performance (losses for training and validation sets
fig, axes = plt.subplots(ncols=3, figsize=(20,5))

axes[0].plot(hist.history['total_loss'], color='teal', label='loss')
axes[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
axes[0].title.set_text('Loss')
axes[0].legend()

axes[1].plot(hist.history['class_loss'], color='teal', label='class loss')
axes[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
axes[1].title.set_text('Classification Loss')
axes[1].legend()

axes[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
axes[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
axes[2].title.set_text('Regression Loss')
axes[2].legend()

plt.show()

"""11. Make Predictions"""

test_samples = test_set.as_numpy_iterator()

next_testsample = test_samples.next()

pred = facetracker.predict(next_testsample[0]) # Run prediction

# Plot predictions from testing set
fig, axes = plt.subplots(ncols=4, figsize=(20,20))
for i in range(4):
    img = test_sample[0][i]
    bbox = pred[1][i]

    if pred[0][i] > 0.9:
        img_copy = img.copy()
        cv2.rectangle(img_copy,
                      tuple(np.multiply(bbox[:2], [120,120]).astype(int)),
                      tuple(np.multiply(bbox[2:], [120,120]).astype(int)),
                            (255,0,0), 2)

    axes[i].imshow(img_copy)

from tensorflow.keras.models import load_model

facetracker.save('facetracker.h5') # Save model

facetracker = load_model('facetracker.h5') # Load model

facetracker.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Compile model

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Iterate through the test dataset and compare actual and predicted results
def evaluate_model(test_dataset, model, threshold=0.5):
    y_true_list, y_pred_list = [], []

    # Iterate through batches in the test dataset
    for batch in test_dataset.as_numpy_iterator():
        X, y = batch
        predictions = model.predict(X)

        # Get class label for actual and test images
        y_pred_batch = (predictions[0] > threshold).astype(int).flatten()  # Predicted classes
        y_true_batch = y[0].flatten()  # Actual classes

        # Add values to lists
        y_true_list.extend(y_true_batch)
        y_pred_list.extend(y_pred_batch)

    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)

    # Calculate performance metrics
    accuracy = accuracy_score(y_true_array, y_pred_array)
    f1 = f1_score(y_true_array, y_pred_array)
    precision = precision_score(y_true_array, y_pred_array)
    recall = recall_score(y_true_array, y_pred_array)
    conf_matrix = confusion_matrix(y_true_array, y_pred_array)

    # Print performance metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Face", "Face"], yticklabels=["Non-Face", "Face"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Generate and evalute predictions on test dataset
evaluate_model(test, facetracker)

# Real-Time Detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read() # Read frame
    frame = frame[50:500, 50:500,:]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert color method
    resize = tf.image.resize(rgb, (120,120)) # Resize to fit data pipeline format

    pred = facetracker.predict(np.expand_dims(resize/255,0))
    bbox = pred[1][0]

    if pred[0] > 0.5:
        # Draw bounding box
        cv2.rectangle(frame,
                      tuple(np.multiply(bbox[:2], [400,400]).astype(int)),
                      tuple(np.multiply(bbox[2:], [400,400]).astype(int)),
                            (255,0,0), 2)
        # Area to print bounding box label
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(bbox[:2], [400,400]).astype(int),
                                    [0,-30])),
                      tuple(np.add(np.multiply(bbox[:2], [400,400]).astype(int),
                                    [80,0])),
                            (255,0,0), -1)

        # Add label to bounding box
        cv2.putText(frame, 'Raj', tuple(np.add(np.multiply(sample_coords[:2], [400,400]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('FaceDetection', frame) # Display output feed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

