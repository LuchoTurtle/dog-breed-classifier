# Dog Breed Classifier
This is the second project of the Udacity Deep Learning nanodegree, part of my voyage to have a grasp of what the hell is going on with Deep Learning and Machine Learning in general :smile:.

In this notebook, I'm attempting to predict a dog breed based on an image by creating a CNN, both from scratch and from a pre-trained model (in this case, I'm using the ResNet50 one) and comparing results.  It is implemented using PyTorch.

This project is particularly interesting because it encompasses much of the pipeline of training a model, from data augmentation to data visualization and hyper parameter tweaking.

## Deployment
Within this repo, there is an `app` folder that has the same features as found in the notebook, albeit with some differentes. You should check the [HTML file](./app/dog_app2.html) to check the progress and added features. The `classifier.py` file has the prediction method reordered, knowing if there is a dog or cat through an ensemble of two ResNet models, averaging the prediction of each model to reach to a solid conclusion on if the dog pertains to a dog or a cat.

 It uses Streamlit to create a simple frontend app that the users can use the classifier. It's better than just having a notebook, right? :sweat_smile:

The code that is in the `app` folder is the same as found in the notebook, with an additional feature of detecting if it is a cat.

> The way it detects the cat is in the same way it detects a dog: it uses the pre-trained VGG-16 model and checks the index of the prediction. There are a few classes that relates to cats so if it predicts any of these, we can consider that the image showed is a cat.

The `app` folder is has all the files needed to easily deploy to Heroku. If you want to do so, you can follow the instructions [here](https://towardsdatascience.com/from-streamlit-to-heroku-62a655b7319) .


## License
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
