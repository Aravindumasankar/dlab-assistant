'''
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The recognition classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the '_train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-_train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

'''

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face
import cv2
from face.face_recognition_cli import image_files_in_folder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import numpy as np

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('face/emotion/model/model.h5')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir='train', model_save_path='model', n_neighbors=None, knn_algo='ball_tree', verbose=True):
    for class_dir in os.listdir(train_dir):
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print(img_path)
            # try to load the image from disk
            image = cv2.imread(img_path)
            # if the image is `None` then we could not properly load the
            # image from disk (so it should be ignored)
            if image is None:
                print("[INFO] deleting: {}".format(img_path))
                os.remove(img_path)
    X = []
    y = []
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print(img_path)
            image = face.load_image_file(img_path)
            face_bounding_boxes = face.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print('Image {} not suitable for training: {}'.format(img_path, "Didn't find a face'" if len(
                        face_bounding_boxes) < 1 else "'Found more than one face"))
                    os.unlink(img_path)
            else:
                # Add face encoding for current image to the training set
                X.append(face.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print('Chose n_neighbors automatically:', n_neighbors)

    # Create and _train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(root_url, filename, X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
    file_path = X_img_path
    data = {}
    pil_image = Image.open(X_img_path).convert('RGB')
    '''
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a recognition classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled recognition classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    '''
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception('Invalid image path: {}'.format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception('Must supply recognition classifier either through knn_clf or model_path')

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    count = 0
    X_img = face.load_image_file(X_img_path)
    X_face_locations = face.face_locations(X_img)


    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        data['faces'] = 0
    else:

        # Find encodings for faces in the test image
        faces_encodings = face.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = [(pred, loc) if rec else ('unknown', loc) for pred, loc, rec in
                       zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        data['model_path'] = model_path
        data['faces'] = len(predictions)
        data['predictions'] = predictionsDump(root_url,filename, file_path, predictions)
    return data


def predictionsDump(root_url,filename, img_path, predictions):
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    data = {}
    data['extracted_images'] = []
    '''
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    '''
    pil_image = Image.open(img_path).convert('RGB')
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw = ImageDraw.Draw(pil_image)
    data['extracted_images'] = []
    # x - left
    # y - upper
    # w right
    # h botton
    for name, (top, right, bottom, left) in predictions:
        cv2.rectangle(image, (left, top - 50), (left + right, top + bottom + 10), (255, 0, 0), 2)
        roi_gray = gray[top:top + bottom, left:left + right]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = str(emotion_dict[maxindex])
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        extracted_images = pil_image.crop((left, top, right, bottom))
        # Save in trained directory
        if name != 'unknown':
            if not os.path.exists('known_people/' + name):
                os.makedirs('known_people/' + name)
            extracted_images.save('known_people/' + name + '/' + str(top) + '---' + filename, 'JPEG')
            data['extracted_images'].append(
                {"name": name, "coordinates":[top,left,bottom,right],"path": root_url+'known_people/' + name + '/' + str(top) + '---' + filename, 'emotion': emotion})
        else:
            if not os.path.exists('unknown_people/'):
                os.makedirs('unknown_people/')
            extracted_images.save('unknown_people/' + str(top) + '---' + filename, 'JPEG')
            data['extracted_images'].append(
                {"name": name, "coordinates":[top,left,bottom,right],"path": root_url+'unknown_people/' + str(top) + '---' + filename, 'emotion':emotion})
        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode('UTF-8')

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    # Retrain Model
    # data['model_info'] = init(TODAY_MODEL)

    # Display the resulting image
    if not os.path.exists('processed/'):
        print("New directory created")
        os.makedirs('processed/')
    pil_image.save('processed/' + filename, 'JPEG')
    data['processed_image_file'] = root_url+'processed/'+filename
    return data

if __name__ == "__main__":
    today_model = 'model/' + 'today_model' + '_trained_knn_model.clf'
    train('train', today_model,2)