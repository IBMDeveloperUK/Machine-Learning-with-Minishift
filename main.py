import tensorflow as tf
import os.path

import server

# Check if pre-trained model already exists
if not os.path.exists('mnist.h5'):
    import train

    train.start()

    print('Training complete. Starting server')
    server.start()

else:
    print('Model exists. Starting server')
    server.start()

