from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Flatten, Dense, Input

from keras_vggface.vggface import VGGFace
from keras_vggface import utils

class ModelBase(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X, y=None):
        pass


class VGGFaceModel(ModelBase):
    def __init__(self, verbose_msg=False):
        self._model = VGGFaceModel.__construct_model()

        assert isinstance(verbose_msg, bool)
        self._verbose_msg = verbose_msg

    @staticmethod
    def __construct_model():
        vgg_model = VGGFace()
        return vgg_model
    
    @staticmethod
    def __construct_customized_model():
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output

        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)

        out = Dense(nb_class, activation='softmax', name='fc8')(x)

        custom_vgg_model = Model(vgg_model.input, out)

        vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
        vgg_features.predict()

        return custom_vgg_model

    def fit(self, X, y):
        self._model = VGGFaceModel.__construct_customized_model()
        self._model.fit(X, y)

    def predict(self, X, y=None):
        preds = self._model.predict(X)
        if self._verbose_msg:
            print('Predicted: ', utils.decode_predictions(preds))
        return preds