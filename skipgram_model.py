#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Embedding, Dot
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

class SkipGramModel ():
    def __init__(self, embedding_id, train_students, valid_students, 
                optimizer=Adagrad(learning_rate=0.02879), embed_dim=218, simple=True):
        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.simple = simple
        self.embedding_id = embedding_id
        self.train_students = train_students
        self.valid_students = valid_students

        self.model = self.__build_model()

    def __build_model(self):
        input_course_ = Input(shape=(1,), name='Course_ids')
        input_context_ = Input(shape=(1,), name='Context')
        embed = Embedding(len(self.embedding_id), self.embed_dim, name='Course_embedding')(input_course_)
        embed2 = Embedding(len(self.embedding_id), self.embed_dim, name='Context_embedding')(input_context_)
        output = Dot(-1)([embed, embed2])
        sigmoid = keras.activations.sigmoid(output)

        self.model = Model(inputs=[input_course_, input_context_],outputs=sigmoid, name='Model')
        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])

        return self.model

    def train(self, data, filepath=None):
        if filepath is not None:
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
        else:
            callbacks_list = None
        self.model.fit(data.train_generator(), validation_data=data.valid_generator(), steps_per_epoch = len(self.train_students),
                validation_steps = len(self.valid_students), epochs = 150)

    def get_embedding(self):
        return self.model.layers[2].get_weights()[0]

    def save(self, fname):
        self.model.save_weights(os.path.join('skip_gram_weights', fname))

    def load(self, fname):
        path = os.path.join('skip_gram_weights', fname)
        self.model.load_weights(path, by_name=True)
