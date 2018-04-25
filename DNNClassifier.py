### Heartbeat Audio Classification using tf.estimator.DNNClassifier

# from vggish_input import wavfile_to_examples
from sklearn.model_selection import train_test_split

import glob
import argparse
import numpy as np
import tensorflow as tf
assert tf.__version__ == "1.7.0"
tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = None


def main():
    if FLAGS.dataset is None:
        raise ValueError("You must pass a path to a dataset")

    files_and_labels = {path: path.split('/')[5] for path in glob.glob(FLAGS.dataset + "**/*.wav")}

    x, y = [], []
    for wav in files_and_labels:
        data = wavfile_to_examples(wav)  # log mel spectrogram examples
        if len(data) != 0:
            x.append(data[0])
            y.append(files_and_labels[wav])  # the class of the wav file

    x = np.array(x)
    y = np.array(y)

    # Let's split the data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    def input_fn(x, y, batch_size=32, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.numpy_input_fn(
            x={'x': x},
            y=y,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle
        )

    # Define the feature column x. x is a [num_examples, 96, 64] matrix
    feature_col_x = tf.feature_column.numeric_column('x', shape=(96,64))

    estimator = tf.estimator.DNNClassifier(
        n_classes=5,
        model_dir=FLAGS.model_dir,
        label_vocabulary=['normal',
                          'noisy_normal',
                          'murmur','extrastole',
                          'noisy_murmur'],
        feature_columns=[feature_col_x],
        hidden_units=[512, 256, 128],
        optimizer=tf.train.AdamOptimizer(
            learning_rate=0.001
        ),
        activation_fn=tf.nn.relu,
        dropout=None,
        loss_reduction=tf.losses.Reduction.SUM
    )

    estimator.train(input_fn=input_fn(
        x=x_train,
        y=y_train,
        num_epochs=None,
        shuffle=True), steps=FLAGS.steps or 1200)

    estimator.evaluate(input_fn=input_fn(
        x=x_test,
        y=y_test,
        num_epochs=1,
        shuffle=False
    ), steps=FLAGS.steps or 150)

    if FLAGS.predict:
        predictions = estimator.predict(input_fn=input_fn(x=x_test,
                                                          y=None,
                                                          batch_size=50,
                                                          num_epochs=1,
                                                          shuffle=False))
        expected = y_test

        for pred_dict, expec in zip(predictions, expected):
            pred_class = pred_dict['classes'][0].decode('utf-8')  # Choose the predicted class with the highest probability
            prob = pred_dict['probabilities'][0] * 100

            if pred_class != expec:
                print("Predicted: `{0}` ({1:0.3f}%), Actual: `{2}` \t WRONG".format(pred_class, prob, expec))
            else:
                print("Predicted: `{0}` ({1:0.3f}%), Actual: `{2}`".format(pred_class, prob, expec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict',
        type=bool,
        default=True,
        help='Set to true to predict on testing data'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps to train model'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Path to save(d) model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset'
    )
    FLAGS, unparsed = parser.parse_known_args()

    main()
