#Image Caption Generator
##RNN and Markov Chain-based Caption Generation

This is the result of me being bored at 3 AM during exam week when I can't sleep because REM cycles or something.

Anyway, this is just a rough sketch of how captions can be generated from just an image.

##How it works
First, the image gets processed through Clarifai and returns tags that describe the image. I then took [hunkim](http://github.com/hunkim)'s word-RNN repository to take a set of encouraging messages and generate a heuristic of a caption based on the relevant captions.

##Installation
###Step 1:
Install the [Clarifai Python api](https://github.com/Clarifai/clarifai-python). Set it up according to their instructions, including getting your own Client ID and Secret.

###Step 2:
Install the remaining dependencies, including `markovify`, `numpy`, and `tensorflow` if you don't have them already.

###Step 3:
Change line 33 of `index.py` to the URL of an image you want to caption.