# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np

import tensorflow as tf
import cv2
from sklearn.utils import shuffle

# Parameters
learning_rate = 0.001
training_iters = 400000
batch_size = 128
display_step = 100

n_input = 1024
n_classes = 43
num_channels = 1

# Image Preprocesing
# We will create more training examples from the data by applying pixel manipulation and image rotation

def do_rotate(img, angle, col_size, row_size):
    M = cv2.getRotationMatrix2D((col_size / 2, row_size / 2), angle, 1)
    return cv2.warpAffine(img, M, (col_size, row_size))


def do_pixel_shift(img, x_shift, y_shift, row_size, col_size):
    M = np.float32([[1, 0, x_shift],[0, 1, y_shift]])    
    return cv2.warpAffine(img, M, (col_size, row_size))


def do_yuvscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def do_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Training Data Generation
def generate_random_batch(training_data, training_labels, size):
    idx = np.random.choice(training_data.shape[0], size)
    return training_data[idx], training_labels[idx]


X_train = np.array([do_grayscale(x) for x in X_train])
X_test = np.array([do_grayscale(x) for x in X_test])
training_image_rowsize = X_train[0].shape[0]
training_image_colsize = X_train[0].shape[1]


rotated_X_train = np.array([do_rotate(x,
                                      np.random.randint(low=-15, high=15),
                                      training_image_colsize,
                                      training_image_rowsize) 
                            for x in X_train])


shifted_X_train = np.array([do_pixel_shift(x,
                                           np.random.randint(low=-2, high=2),
                                           np.random.randint(low=-2, high=2),
                                           training_image_colsize,
                                           training_image_rowsize) 
                            for x in X_train])

X_train = np.concatenate((X_train,np.concatenate((rotated_X_train, shifted_X_train), axis=0)), axis=0)
y_train = np.concatenate((y_train, np.concatenate((y_train, y_train), axis=0)), axis=0)
X_train = X_train.reshape((-1, n_input)).astype('float32')
X_train, y_train = shuffle(X_train, y_train, random_state=0)

# Convert class values into a one hot encoded vector
n_values = (np.max(y_train) + 1)
y_train_ohe = np.eye(n_values)[y_train.astype('int')]
n_values = (np.max(y_test) + 1)
y_test_ohe = np.eye(n_values)[y_test.astype('int')]

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.int8, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 32, 32, num_channels])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)
        conv1 = tf.nn.local_response_normalization(conv1)
        
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)
        conv2 = tf.nn.local_response_normalization(conv2)
        
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, num_channels, 128], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([8*8*128, 256], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([256, n_classes], stddev=0.1))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([256], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    }


    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = generate_random_batch(X_train, y_train_ohe, batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, 
                                       y: batch_y,
                                       keep_prob: .75})
        if (step * batch_size) % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, 
                                                        y: batch_y,
                                                        keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)
    
    total = 0
    for i in range(12):
        batch_x, batch_y = X_test.reshape((-1, n_input))[i*1000:(i+1)*1000], y_test_ohe[i*1000:(i+1)*1000]
        total += sess.run(accuracy, feed_dict={x: batch_x,y: batch_y, keep_prob: 1.})
    print ("Total accuracy", total / 12)