import os
import sys
import site
sys.path.append(os.getcwd())

import cv2
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
from mlbox.models.ml_models import VGGFaceModel

class FaceRecognitionDefaultVGG(object):
    def __init__(self):
        # Models for Face recognition
        self._model = VGGFaceModel(verbose_msg=True)

        # Models for face detection
        cv2_haarcascade_config = '%s%scv2%sdata%shaarcascade_frontalface_default.xml' % (site.getsitepackages()[0], os.sep, os.sep, os.sep)
        self._face_detector = cv2.CascadeClassifier(cv2_haarcascade_config)

    def __recognize_face(self, face_detected):
        x = cv2.resize(face_detected, (224, 224))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        X = np.expand_dims(image.img_to_array(x), axis=0)
        X = utils.preprocess_input(X, version=1) # or version=2

        return self._model.predict(X)

    def worker(self):
        cap = cv2.VideoCapture(0)

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects_face_detected = self._face_detector.detectMultiScale(frame_gray, 1.3, 5)
            for (x, y, w, h) in rects_face_detected:
                self.__recognize_face(frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('FRAME_ORIGINAL', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = FaceRecognitionDefaultVGG()
    detector.worker()