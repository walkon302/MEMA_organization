import numpy as np
import tensorflow as tf
import csv
import os

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
  # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 128, 128, 1])

  # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_model(train_sample, train_label, training_step):

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=10)

    mema_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/Users/Walkon302/Desktop/MEMA_organization/MEMA_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_sample},
        y=train_label,
        batch_size=5,
        num_epochs=None,
        shuffle=True)

    mema_classifier.train(
        input_fn=train_input_fn,
        steps=training_step,
        hooks=[logging_hook])

def eval_model(eval_sample, eval_label):

    mema_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/Users/Walkon302/Desktop/MEMA_organization/MEMA_model")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_sample},
        y=eval_label,
        num_epochs=10,
        shuffle=False)

    return mema_classifier.evaluate(input_fn=eval_input_fn)

def pred_model(pred_sample):

    mema_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/Users/Walkon302/Desktop/MEMA_organization/MEMA_model")

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": pred_sample},
        num_epochs=1,
        shuffle=False)

    return mema_classifier.predict(input_fn=pred_input_fn)

def prediction(prediction, file_name):
    result = []
    for pred, name in zip(prediction, file_name):
        if pred['classes'] == 0:
            result.append((name, 'organized'))
        else:
            result.append((name, 'disorganized'))

    curdir = os.path.dirname(os.getcwd())
    directory = '{}/output'.format(curdir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = '{}/output.csv'.format(directory)

    with open(output_file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for r in result:
            writer.writerow([r])
