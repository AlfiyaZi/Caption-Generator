#Import useful packages
import numpy as np
import tensorflow as tf
import markovify

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from clarifai.rest import ClarifaiApp

#Generate word-RNN-based text 
def sample(search):
    with open(os.path.join('save', 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join('save', 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('save')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess, words, vocab, 50, search, 1, 1, 4)

#Get tags from Clarifai
app = ClarifaiApp()
data = app.tag_urls(['https://static.pexels.com/photos/126407/pexels-photo-126407.jpeg'])
d = [a['name'] for a in data['outputs'][0]['data']['concepts']]

text_from_rnn = ""
for a in d:
	try:
		s = str(sample(a))
		text_from_rnn += s+'\n'
	except Exception:
		s = ""
text_from_rnn += s+'\n'
print text_from_rnn

orig_text = open('data/messages/input.txt').read()

model_a = markovify.Text(text_from_rnn)
model_b = markovify.Text(orig_text)

#Combine Markov chains from original text and the RNN text
model_combo = markovify.combine([ model_a, model_b ], [ 1, 1.5 ])

#Generate five captions
for a in range(1, 6):
	print "Caption "+str(a)+": "+model_combo.make_sentence()