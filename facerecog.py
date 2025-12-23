from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class FacialExpressionRecognizer:
    def __init__(self, train_dir, test_dir, batch_size=48, epochs=40):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
        self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        
    def prepare_data(self):
        traindatagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            shear_range=0.3,
            zoom_range=0.3,
            width_shift_range=0.4,
            height_shift_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validdatagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = traindatagen.flow_from_directory(
            self.train_dir,
            color_mode='grayscale',
            target_size=(48, 48),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.validation_generator = validdatagen.flow_from_directory(
            self.test_dir,
            color_mode='grayscale',
            target_size=(48, 48),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print(self.model.summary())

    def count_images(self, path):
        num_imgs = 0
        for root, dirs, files in os.walk(path):
            num_imgs += len(files)
        return num_imgs

    def train(self):
        num_train_imgs = self.count_images(self.train_dir)
        num_test_imgs = self.count_images(self.test_dir)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=self.epochs,
            validation_data=self.validation_generator,
            validation_steps=len(self.validation_generator),
            callbacks=[early_stop, reduce_lr]
        )

    def save_model(self, filename='facial_expression_model.h5'):
        self.model.save(filename)

    def loadall(self):
        if os.path.exists('facial_expression_model.h5'):
            print("File Already Exists")
            if input("Enter 't' to test or 'd' to deploy: ").lower() == 't':
                self.test()
            else:
                self.deployface()
        else:
            self.prepare_data()
            self.build_model()
            self.train()
            self.save_model()

    def deployface(self):
        self.model = load_model('facial_expression_model.h5')
        vid = cv2.VideoCapture(0)

        while True:
            ret,frame = vid.read()
            graycolour = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_haar_cascade.detectMultiScale(graycolour, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                subfaceimg = graycolour[y:y + h, x:x + w]
                resizeimage = cv2.resize(subfaceimg, (48, 48))
                normalizeimg = resizeimage / 255.0
                reshapedimg = np.reshape(normalizeimg, (1, 48, 48, 1))
                result = self.model.predict(reshapedimg)
                label = np.argmax(result, axis=1)[0]
                print(label)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,50,50),2)
                cv2.rectangle(frame,(x,y-40),(x+w,y),(255,50,50),-1)
                cv2.putText(frame, self.emotion_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            cv2.imshow('Facial Emotion Recognition', frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()            

train_path = 'kagglehub/datasets/msambare/fer2013/versions/1/train'
test_path = 'kagglehub/datasets/msambare/fer2013/versions/1/test'
