# Requirements
I used python 3.8, earlier versions may not be compatible
Requires Nvidia GPU with Cuda runtime installed

The train and test image sets should be placed int hte cats_vs_dogs directory. 
They are available here:  https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

# Instructions
If on windows, run installDependencies.bat in cmd to install the necessary python modules via pip
Otherwise open that batch file and run the commands contained in there individually in your prefered shell environment.

Note: When the running the program for the first time, you should edit DogsCats.py and set the boolean flags 'regen_data' and 'train_model' both to true, so that the model is regenerated locally on your machine. Just in case, I have included a default model in case you forget this step. 

Then run the following command: python DogsCats.py
Or use Run.bat if on windows

The program will eventually prompt for numbers corresponding to images you would like to test using the trained model.
For example, if you enter 5, the model will examine the file test/5.jpg and predict whether it is a cat or dog.  

# Credits
Some code, particularly the portion where the neural network is generated and trained, was taken from the following YouTube series: https://www.youtube.com/watch?v=gT4F3HGYXf4

# YouTube Demonstration
A video demonstrating the results of this project is available here:
https://youtu.be/4hGCEo9ebCQ
