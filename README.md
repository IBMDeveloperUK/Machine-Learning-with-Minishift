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

## Before we start...

It's probably best that you install MiniShift before we start diving into neural networking goodness. [Mofe Salami](https://twitter.com/Moffusa) has put together a [fantastic workshop](https://github.com/IBMDeveloperUK/minishift101/tree/master/workshop) that walks you through the installation and basic setup of MiniShift. If you pop on over there and follow just the setup steps of the workshop, and then head back here, we'll be good to crack on.

## You will need:

1. A Github account
2. A macOS/Windows/Linux system capable of running MiniShift
3. A modern web browser

## Recognising handwritten digits with Keras + the MNIST Dataset

Training neural networks (NNs) to classify handwritten digits has become something of a "Hello, World" for developers looking to start tinkering with neural networks. The reasons for this are myriad, but three stand out:

1. The dataset is small, so the network can be trained in a short space of time.
2. For a very long time, computers struggled to recognise natural human input, but with NNs the problem is essentially trivial to solve (we'll likely get a > 98% accuracy with the model we'll build)
3. The architecture for recognizing handwritten digits is reuseable for wider image classification cases, so if you're looking to analyse visual datasets with CNNs, MNIST is a great way to cut your teeth.

