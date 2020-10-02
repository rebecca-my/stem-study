import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

class SkipGramModel ():
	def __init__(self, embedding_dim, optimizer, simple=True):
		self.embedding_dim = embedding_dim
		self.optimizer = optimizer
		self.simple = simple

		self.model = self._build_model()

	self.embedding_dim = 100
	self.optimizer = Adam(lr=0.001)

	def _build_model(self):
	    input_ = Input(shape=x.shape[1:], name='Course ids')
	    embed = Embedding(len(embedding_id), self.embedding_dim, name='Course embedding')(input_)
	    output = Dense(len(embedding_id), activation='softmax', name='Course probabilities')(embed) 
	    
	    model = Model(inputs=input_,outputs=output, name='Model')
	    model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['acc'])
	    return model

	def train(data):
	    train_gen = data.train_generator(False)
	    valid_gen = data.valid_generator(False)
	    model.fit_generator(train_generator()
	                        #validation_data=valid_generator(),
	                        steps_per_epoch = data.n_train, 
	                        validation_steps= data.n_valid,
	                        epochs=150,
	                        verbose=1,)

