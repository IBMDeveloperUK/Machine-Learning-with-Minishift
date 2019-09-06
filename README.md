# Training and Deploying Machine Learning Models with Containers

## Machine learning dependencies are a hassle...

Between ensuring that the right version Python/Pip are installed on your system, and that it doesn't conflict with other Python/Pip versions on your system AND that when you deploy your model to the cloud that the versions of the dependencies you've used in your projects are still compatible with the version on your cloud-based system, it's a wonder that we ever get any time to focus on building and training our neural networks.

Fortunately, there's a way to ensure that all of this is a never a problem again - Containers! (specifically, [MiniShift](https://github.com/minishift/minishift) )

With containers, we can create a clean, virtual environment to setup and train our neural networks in, and then deploy them at scale with the _exact same_ same environment. No more dependency hell!

## "But... won't that be slower?"

As with everything in life, there are caveats to this approach. You are training your network on a virtualised system, so you're not going to get the full, raw power of your machine being utilised in the training process. Even with small networks, training can take quite some time, even longer inside a virtual environment. However, if you're a machine learning focussed developer with myriad networks to iterate and train, managing all of those dependencies can take hours to configure, and there's no guarentee that if there isn't a problem on your system, that there won't be when it's deployed to the production environment.

Although this approach will take longer to train, the time savings in reducing the complexity of your setup should work to offset, *and* when you complete this workshop, you'll be able to deploy your model to a super-scalable OpenShift Cluser (if you so wish) where you can scale to meet the needs of your users in next to no time at all.

## "Can't I just use a Virtual Environment instead?"

Absolutely, if that works for you, go for it, but depending on the virtual environment you're using, it can be equally as awkward to prepare your project as managing the dependencies manually (in fact, I had the idea for this workshop after spending 6 hours fighting with my local environment). There's also guarentee that the environment you deploy your application to will have a matching configuration without some pre-emptive tweaking.

## "OK... I'm interested..."

Cracking, then let's get started!

## In this workshop you will learn...

1) How to build a Convolutional Neural Network (CNN) that can detect handwritten digits (with Keras and the MNIST dataset)
2) How to train and deploy a CNN with the Flask web framework and Keras
3) How to install and run MiniShift (a locally run OpenShift cluster of one image) on your machine
4) How to create a project in OpenShift
5) How to create an app in OpenShift and pull the source code for application from Github

By the end, you'll end up with a natty web app that will tell you what characters you're drawing, that'll look like this:

![A video demonstrating the classification web app](/resources/tada.gif)

## Before We Start...

It's probably best that you install MiniShift before we start diving into neural networking goodness. [Mofe Salami](https://twitter.com/Moffusa) has put together a [fantastic workshop](https://github.com/IBMDeveloperUK/minishift101/tree/master/workshop) that walks you through the installation and basic setup of MiniShift. If you pop on over there and follow just the setup steps of the workshop, and then head back here, we'll be good to crack on.

## You Will Need:

1. A Github account
2. A macOS/Windows/Linux system capable of running MiniShift
3. A modern web browser

## Recognising Handwritten Digits with Keras + the MNIST Dataset

Training neural networks (NNs) to classify handwritten digits has become something of a "Hello, World" for developers looking to start tinkering with neural networks. The reasons for this are myriad, but three stand out:

1. The dataset is small, so the network can be trained in a short space of time.
2. For a very long time, computers struggled to recognise natural human input, but with NNs the problem is essentially trivial to solve (we'll likely get a > 98% accuracy with the model we'll build)
3. The architecture for recognizing handwritten digits is reuseable for wider image classification cases, so if you're looking to analyse visual datasets with CNNs, MNIST is a great way to cut your teeth.

## Starting Your Project

The code in this repo is a scaffold for the neural network and app that you'll end up with if you follow this workshop to the end.

So we can get the full benefit of MiniShift's ability to pull code from a centralised repository and deploy it, you'll need to fork this repo to create your own version of it to work from.

You can do that with the following steps

1. If you've not done so already, log in to your Github account (or create one [here](https://github.com/join) .
2. Head back to this repository and then click the fork button at the very top of the UI. It looks like this:

![An image highlighting to fork button](/resources/fork_btn.png)

This will create a copy of this repository that you'll be able to make changes to, and deploy from.

3. Once the forking process has completed, you need to clone it to your local system. You can do this by clicking the green "Clone or download" button just beneath the navigation for your repo, and then copying either the HTTPS or SSH link in the dialog that appears.

![An image highlighting to fork button](/resources/clone.png)
![An image highlighting to fork button](/resources/clone_link.png)

4. Once you've copied either link, head to your terminal and enter: `git clone <URL YOU JUST COPIED>`

5. This will copy your forked version of the project to your local system. Now we're ready to start building a neural network! 🎉

## A Quick Tour

If you take a moment to look at the project you just cloned, you'll see a bunch of files and folders. Here is a brief description of each one

1. `reference` - A complete implementation of the project that we'll be making.
2. `app.py` - The main entry point for our program. When we run our project shortly, MiniShift will look for and execute this script
3. `requirements.txt` - A text file which describes which dependencies our project will need to install to support our program.
4. `server.py` - The code that will serve our prediction web app.
5. `train.py` - The code that contains our neural network which will begin training the first time we run our app.

## Building Our Neural Network

### Importing Dependencies
Before we can worry about training and/or deploying our neural network, we first need to construct it! We'll be using the demo code from the Keras documentation to get us started quickly. You can [head over there](https://keras.io/examples/mnist_cnn/) to read about that (or other examples), or you can stick around here where we'll walk through a few of the blocks of code and explain their function as we work through them.

With your favourite IDE, open the file `train.py`. This is where the code that creates, trains, and stores our model for later deployment will live.

First, we need to import the dependencies that our project will need to run. You don't need to have these installed on your system, MiniShift will take care of this later.

In `train.py` copy and paste the following code block at the top of the file:


```python
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
```

### Preparing our data for training

Next, we'll create a function `start` which our application will call to start training our model when we spin up our container (if a model hasn't already been trained).

```python
def start():

    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
```

The variables `batch_size`, `num_classes`, and `epochs` tell our program how to digit images to load into memory and pass through our network (`batch_size`), how many different types of digits (or classes) there are (`num_classes`), and how many times to pass the entire set through the neural network during the training phase `epochs`.

`img_rows, img_cols = 28, 28` will be used by our neural network to determine the _shape of our data_. Described [here](https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca) as: 

***"[...] input data has a shape of (batch_size, height, width, depth), where the first dimension represents the batch size of the image and other three dimensions represent dimensions of the image which are height, width and depth. For some of you who are wondering what is the depth of the image, it’s nothing but the number of colour channel. For example, RGB image would have a depth of 3 and the greyscale image would have a depth of 1."***

`(x_train, y_train), (x_test, y_test)` are variables that will have the data from the MNISt dataset split up into training and validation sets that will be used by the neural network to confirm whether or not it's getting better at the job we've assigned it.

Beneath that, copy and paste the following code snippet. Be sure to maintain the indentation, as Python is very picky about that sort of thing...

```python
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
```

Depending on the data being processed, some image datasets will have their data stored in a 3-Dimensional array in the following order `[channels][rows][cols]` - This is called _"channel-first"_ data. Conversely, other image datasets will have their information stored `[rows][cols][channels]` - _"channel-last"_. This little snippet of code is just a little bit of sugar that sets up our neural network to receive the information regardless of which way around it is.

The next snippet of code is the final bit of processing of our dataset neccessary to pass it through the network we're about to construct. Copy the following code block and paste it on a new line just after the last code snippet.

```python
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
```

Here, we're first casting all of the values describing each image to floats. Next, we divide those values by 255 (the maximum value for a channel - R, G, or B - in a pixel for an RGB image) so that it's a value between 0 and 1. We then `print` out the data for the first image from our training and test datasets

`y_train` and `y_test` contain the labels of each of the images in our datasets (otherwise our neural network wouldn't know what any of the data in `x_train` or `x_test` actually meant). With `keras.utils.to_categorical(y_train, num_classes)` we're converting the labels to an index in a matrix which maps to the original categorisations of the images.

### Constructing our model

For our project, we're going to create a basic 2-Dimensional Convolutional Neural Network. This architecture has been shown to be very effective at recognising patterns in images, and because it's a sequential model (a model data passes from each layer to the next) it's considered to be more more legible than other similarly tasked networks.

Copy and paste the next code snippet on a new line after the last snippet we added.

```python
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
```

Our first variable `model` is where we'll create a reference that we can use to construct each layer of neurons in our neural network.

Next, we add a 2D Convolutional layer to our network with `model.add(Conv2D(32 kernel_size=(3, 3), activation='relu', input_shape=input_shape))`. This layer "convolves" around our image in an attempt to find patterns and structures that it can use in classifying our images. Adit Deshpande has written a [great blog post](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) where you can read more about what's going on in this layer, if you're so inclined.

We compound this effect with a second convolutional layer `model.add(Conv2D(64, (3, 3), activation='relu'))`.

After that, with `model.add(MaxPooling2D(pool_size=(2, 2)))`, we create a pooling layer, which is akin to downsampling the results of the two previous layers so that only the most prevalent features pass through to the next layer.

Our next layer is a "dropout" layer. This layer of neurons will randomly ignore input as the network is trained. This encourages the network to be more robust, by depending on a variety of connections between the next and previous layers, rather than relying on a we, heavily weighted connections which could possibly have too much of a say in the ultimate classifcation of the image. 

Next, we flatten the input the input from our dropout layer with `model.add(Flatten())` into a 1-Dimensional array. Up until this point, our data has maintained the original shape, albeit modified, from when we passed it through to our network.

The next layer, `model.add(Dense(128, activation='relu'))` a densely connected layer of neurons with a ReLu activation function serves to act as filter for neurons that have haven't received enough input to be activated. ReLu is a permissive activation function, so strong, and medium strength activations will be allowed to pass through and activate neurons in the next layer of our network, whereas weak activations will not. 

Finally, we have two more layers of neurons. Another dropout layer, and another Densely connected layer. The final layer has the same number of outputs as the number of classes that we wish to categorise our images as. Each neuron corresponds to a label for each class, whichever neuron has the highest activation value will be the classification assigned to the image passed through to the network for training / prediction.

### Compiling our model and training

Now that we've constructed our model, it's time to compile it. This is the stage where we tell our neural networks how to measure it's success in categorising a given input, and start passing through data for training

First, we'll copy and paste the code that will compile our model on a new line after out last code snippet.

```python
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
```

And next, we'll start passing through our data with `model.fit()`

```python
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
```

### Saving our model

Once our network has been trained, it's time to save our model so we can use it later. Fortunately, this is a comparatively simple affair.

Copy and paste the following line of code on a new line beneath our last snippet.

```python
    model.save('mnist.h5')
```

When our neural network has finished training, it's knowledge will be saved to a file "mnist.h5" in the same directory as our project.