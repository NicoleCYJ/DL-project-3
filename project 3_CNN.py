import numpy as np
from skimage import io
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_epochs = 10
batch_size = 320
width = height = 100
num_channels = 1
num_classes = 2
drop_out = 0.50
x = tf.placeholder(tf.float32, shape=[None, width, height, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, 1)

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 8
filter_size_conv2 = 3
num_filters_conv2 = 16
filter_size_conv3 = 3
num_filters_conv3 = 8
fc_layer_size = 16


def read_index(path):
    '''
    to read the index
    data have lable
    '''
    path = path + '.txt'
    file = open(path)  #
    try:
        file_context = file.read()
        #  file_context = open(file).read().splitlines()

    finally:
        file.close()
    file_context = file_context.replace('\t', '\n').split('\n')
    if len(file_context) % 2 == 1:
        len_of_context = len(file_context) - 1
    else:
        len_of_context = len(file_context)

    x_index = list([])
    y = list([])
    for i in range(len_of_context):
        if i % 2 == 0:
            x_index.append(file_context[i])
        else:
            y.append(file_context[i])
    return x_index, y


def true_name(all_name, local_name):
    a = False
    for name in all_name:
        if name[0:8] == local_name:
            a = True
            return name, a
    return local_name, a


def isblack(data):
    if sum(sum(data)) < 20:
        return True
    else:
        return False


def get_picth(all_name, x_train_index, patch_size_1, patch_size_2, y):
    x_patch = []
    x_patch_index = []
    x_train_use_or_not = []
    x_patch_to_image = []
    y_patch = []
    image_size = []
    for i in range(len(x_train_index)):
        x_index_local, have_index = true_name(all_name, x_train_index[i])
        if have_index:
            x_index_local = './Dataset_A/data/' + x_index_local
            pic_raw = io.imread(x_index_local)

            d1 = int(pic_raw.shape[0] / patch_size_1)
            d2 = int(pic_raw.shape[1] / patch_size_2)

            for patch_i in range(d1):
                for patch_j in range(d2):
                    picture_patch = pic_raw[patch_i * 100:patch_i * 100 + 100, patch_j * 100:patch_j * 100 + 100]
                    x_patch.append(picture_patch)
                    picure_index = [patch_i, patch_j]
                    x_patch_index.append(picure_index)
                    x_patch_to_image.append(i)
                    y_patch.append(y[i])
                    image_size.append(pic_raw.shape)
                    if (isblack(picture_patch)):
                        x_train_use_or_not.append(False)
                    else:
                        x_train_use_or_not.append(True)

    return x_patch, x_patch_index, x_train_use_or_not, x_patch_to_image, y_patch, image_size


def get_data_in_patch(all_name, x_train_index, patch_size_1, patch_size_2, y):
    # delete black one
    x_patch, x_patch_index, x_train_use, image_index, y2, image_size = get_picth(all_name, x_train_index, patch_size_1,
                                                                                 patch_size_2, y)
    final_x_patch = []
    final_x_patch_index = []
    final_image_index = []
    final_y = []
    final_image_size = []
    for i in range(len(x_train_use)):
        if (x_train_use[i]):
            final_x_patch.append(x_patch[i])
            final_x_patch_index.append(x_patch_index[i])
            final_image_index.append(image_index[i])
            final_y.append(y2[i])
            final_image_size.append(image_size[i])
    return final_x_patch, final_x_patch_index, final_image_index, final_y, final_image_size



def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    print(np.shape(layer))
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(np.shape(layer))
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    # Get the shape of the layer from the previous layer.
    layer_shape = layer.get_shape()
    # Number of features will be img_height * img_width* num_channels.
    num_features = layer_shape[1:4].num_elements()
    # Flatten the layer
    layer = tf.reshape(layer, [-1, num_features])
    print(np.shape(layer))
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.
    layer = tf.matmul(input, weights) + biases
    print(np.shape(layer))
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
# Apply Dropout
layer_conv1 = tf.nn.dropout(layer_conv1, drop_out)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)
# Apply Dropout
layer_conv2 = tf.nn.dropout(layer_conv2, drop_out)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)
# Apply Dropout
layer_conv3 = tf.nn.dropout(layer_conv3, drop_out)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, 1)


session = tf.Session()

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())


if __name__ == '__main__':
    patch_size_1 = 100
    patch_size_2 = 100

    all_name = os.listdir('./Dataset_A/data')

    x_train_index, y_train = read_index("./Dataset_A/train")
    y_train = np.asarray(y_train, np.int32)

    x_train_patch, x_train_patch_index, image_index, y_train_patch, final_image_size \
        = get_data_in_patch(all_name, x_train_index, patch_size_1, patch_size_2, y_train)

    x_train_patch = np.asarray(x_train_patch)
    x_train_patch = x_train_patch.reshape((-1, 100, 100, 1))

    y_list = []
    for label in y_train_patch:
        if label == 1:
            y_list.append([0, 1])
        elif label == 0:
            y_list.append([1, 0])

    random.shuffle([x_train_patch, y_list])

    total_batches_tr = int(len(y_train_patch) / batch_size) + 1
    print(total_batches_tr)
    for epoch in range(total_epochs):
        acc_tr = 0
        for batch in range(total_batches_tr):
            train_batch = x_train_patch[batch * batch_size:(batch + 1) * batch_size]
            trainLabel_batch = y_list[batch * batch_size:(batch + 1) * batch_size]

            feed_tr = {x: train_batch, y_true: trainLabel_batch}
            # print(session.run(y_pred, feed_dict=feed_tr))
            session.run(optimizer, feed_dict=feed_tr)
            acc = session.run(accuracy, feed_dict=feed_tr)
            acc_tr += acc

            msg = "Training Epoch {0}  Batch {1} ----- Training Accuracy: {2:>6.1%}"
            print(msg.format(epoch + 1, batch + 1, acc))

        acc_tr_avg = acc_tr / total_batches_tr
        msg = "Training Epoch {0}---Training Accuracy: {1:>6.1%}"
        print(msg.format(epoch + 1, acc_tr_avg))

        saver = tf.train.Saver()
        saver.save(session, "./CNN_2")
    session.close()






