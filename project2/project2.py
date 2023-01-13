"""
File: project2.py
Description: CSCI 635 Project 2
Language: python3.8
Author1: Krystian Derhak   kad4374 @rit.edu
Author2: Michael Lee       ml3406@rit.edu
"""

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def read_data(filename, size, model_index):
    """
    The function reads the NSynth dataset and save them into corresponding variables.

    :param filename: the file pathname of the dataset to be loaded
    :param size: the size of the dataset
    :param model_index: an index that indicates which model to train
    :return: the inputs and labels of the dataset
    """
    # read dataset
    dataset = tf.data.TFRecordDataset(filename)

    # map feature description into tensors
    dataset = dataset.map(map_function)

    # the variables to store the data (inputs and outputs)
    examples = list()
    labels = list()

    # a 1D average pool layer used to reduce the input size to 2000
    avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=8, strides=8, padding='valid')

    # a counter used to show loading progress
    i = 1
    # save inputs and labels into corresponding variables
    for example in dataset.take(size):
        # ignore examples of synth lead
        if example["instrument_family"] != 9:
            # keep only the first second
            tensor = tf.slice(example["audio"], [0], [16000])
            tensor = tf.reshape(tensor, [1, 16000, 1])
            tensor = avg_pool_1d(tensor)
            # reshape to [100, 20]
            tensor = tf.reshape(tensor, [100, -1])
            examples.append(tensor)

            # classifier for acoustic, electronic, and synthesized
            if model_index == 3:
                labels.append(example["instrument_source"])
            # classifier for ten instruments
            else:
                # make what were labeled 10 labeled 9 now (or you will get nan loss while training)
                if example["instrument_family"] == 10:
                    labels.append(9)
                else:
                    labels.append(example["instrument_family"])
        print("\r", end="")
        print(str(i) + "/" + str(size), end="")
        i += 1
    print("\r", end="")

    # convert to tensor
    # examples = tf.Variable(np.array(examples))
    examples = np.array(examples)

    # convert to ndarray (convenient for checking shape)
    labels = np.array(labels).astype(np.uint8)

    return examples, labels


def map_function(raw_audio_record):
    """
    The map function of the Nsynth dataset.

    :return: tensor with feature descriptions
    """
    feature_description = {
        'note': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'note_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'instrument': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'pitch': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'velocity': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'sample_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'audio': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
        'qualities': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'qualities_str': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
        'instrument_family': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_family_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'instrument_source': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_source_str': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    return tf.io.parse_single_example(raw_audio_record, feature_description)


def train_model(train_x, train_y, valid_x, valid_y, model_index):
    """
    This function creates a CNN model and trains it with
    a learning rate 0.001, batch size 1024, and epochs 30.

    :param train_x: inputs of training dataset
    :param train_y: outputs of training dataset
    :param valid_x: inputs of validation dataset
    :param valid_y: outputs of validation dataset
    :param model_index: an index that indicates which model to train
    :return: the trained model and the history of training the model
    """

    # construct my model
    my_model = tf.keras.models.Sequential()

    # our LSTM for instrument classifier
    if model_index == 1:
        my_model.add(tf.keras.layers.LSTM(256, input_shape=train.shape[1:], return_sequences=True))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.LSTM(256))
        my_model.add(tf.keras.layers.Dense(128, activation='relu'))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.Dense(64, activation='relu'))
        my_model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # our BLSTM for instrument classifier
    elif model_index == 2:
        my_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True),
                                                   input_shape=train.shape[1:]))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
        my_model.add(tf.keras.layers.Dense(128, activation='relu'))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.Dense(64, activation='relu'))
        my_model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # our model for the bonus
    else:
        my_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True),
                                                   input_shape=train.shape[1:]))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
        my_model.add(tf.keras.layers.Dense(256, activation='relu'))
        my_model.add(tf.keras.layers.Dropout(rate=0.2))
        my_model.add(tf.keras.layers.Dense(64, activation='relu'))
        my_model.add(tf.keras.layers.Dense(3, activation='softmax'))

    my_model.summary()

    # train our model
    '''
    optimizer: Adam optimizer
    loss: cross entropy
    '''
    my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=['accuracy'])

    # use training and validation sets to train our model with batch size = 1024 and epochs = 30
    return my_model, my_model.fit(train_x, train_y, batch_size=1024, epochs=30, validation_data=(valid_x, valid_y),
                                  shuffle=True)


def evaluate_model(my_model, test_x, test_y):
    """
    This function evaluates the trained model using the testing set.

    :param my_model: our trained model
    :param test_x: inputs of the testing dataset
    :param test_y: outputs of the testing dataset
    """

    # evaluate our model
    my_model.evaluate(test_x, test_y)


def make_curves(history):
    """
    This function draws learning curve plots of the trained model.

    :param history: the history of training the model
    """

    # draw learning curve (Cross-entropy V.S. Epoch)
    plt.plot(history["loss"], color=(0.75, 0.75, 0.0), alpha=0.9)
    plt.plot(history["val_loss"], color=(0.0, 0.75, 0.75), alpha=0.9)
    plt.title("Cross-entropy V.S. Epoch", size=18)
    plt.xlim(0, 30)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend(["training", "Validation"])
    plt.show()
    # plt.savefig("Curve_Loss-Epoch.png")
    plt.close()

    # Accuracy V.S. Epoch
    plt.plot(history["accuracy"], color=(0.75, 0.75, 0.0), alpha=0.9)
    plt.plot(history["val_accuracy"], color=(0.0, 0.75, 0.75), alpha=0.9)
    plt.title("Accuracy V.S. Epoch", size=18)
    plt.xlim(0, 30)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["training", "Validation"])
    plt.show()
    # plt.savefig("Curve_Accuracy-Epoch.png")
    plt.close()


def make_plots(my_model, test_x, test_y):
    """
    This function makes plots of the model's confusion matrix, and examples of misclassified samples.

    :param my_model: keras model that has been trained
    :param test_x: test data waveforms (tensor)
    :param test_y: list of labels for the test data (list)
    """

    # Reshapes data so we can use it
    reshape = np.array(test_x)

    # Prime example of one wave
    cleanwave = reshape[3995].reshape(-1)
    cleanwave = cleanwave[0:750]
    plt.figure(20)
    plt.plot(cleanwave)
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('1D Example')
    # plt.savefig('1DExample.png')
    plt.show()

    examples = []
    # 2D example of each wave
    # Loops through an finds the first example of each index
    for x in range(10):
        index = 0
        flag = False
        for i in range(len(test_y)):
            if test_y[i] == x and flag is False:
                ex = reshape[i].reshape(-1)
                ex = ex[0:1500]
                examples.append(ex)
                flag = True
            index += 1

    # Plots the examples
    plt.figure(1)
    plt.plot(examples[1], 'tab:orange', label='Brass')  # Brass
    plt.plot(examples[2], 'tab:green', label='Flute')  # Flute
    plt.plot(examples[3], 'tab:red', label='Guitar')  # Guitar
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms')
    plt.legend()
    # plt.savefig('nn1_class_example_1.png')
    plt.show()
    plt.figure(2)
    plt.plot(examples[4], 'tab:purple', label='Keyboard')  # Keyboard
    plt.plot(examples[5], 'tab:brown', label='Mallet')  # Mallet
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms')
    plt.legend()
    # plt.savefig('nn1_class_example_2.png')
    plt.show()
    plt.figure(3)
    plt.plot(examples[7], 'tab:gray', label='Reed')  # Reed
    plt.plot(examples[8], 'tab:olive', label='String')  # String
    plt.plot(examples[9], 'tab:cyan', label='Vocal')  # Vocal
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms')
    plt.legend()
    # plt.savefig('nn1_class_example_3.png')
    plt.show()
    plt.figure(4)
    plt.plot(examples[6], 'tab:pink', label='Organ')  # Organ
    plt.plot(examples[0], 'tab:blue', label='Bass')  # Bass
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms')
    plt.legend()
    # plt.savefig('nn1_class_example_4.png')
    plt.show()

    # Confusion Matrix
    # pred = reconstructed_model.predict(test_x)
    pred = my_model.predict(test_x)
    pred2 = np.argmax(pred, axis=1)

    cm = confusion_matrix(test_y, pred2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard', 'Mallet', 'Organ',
                                                  'Reed', 'String', 'Vocal'])
    '''disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Acoustic', 'Electronic', 'Synthetic'])'''

    disp.plot()
    plt.title("Confusion Matrix Network")
    # plt.savefig('confusion_matrix.png')
    plt.show()

    # High probability example of each wave
    high_prob_ex = {}
    # Loops through an finds the first example of each index and probability is high
    for x in range(10):
        index = 0
        flag = False
        for k, prediction in zip(range(len(test_y)), pred):
            if test_y[k] == x and prediction[test_y[k]] > 0.5 and flag is False:
                ex = reshape[k].reshape(-1)
                ex = ex[0:1600]
                high_prob_ex[x] = ex
                flag = True
            index += 1
    # Plots the examples
    plt.figure(5)
    plt.plot(high_prob_ex[1], 'tab:orange', label='Brass')  # Brass
    plt.plot(high_prob_ex[2], 'tab:green', label='Flute')  # Flute
    plt.plot(high_prob_ex[3], 'tab:red', label='Guitar')  # Guitar
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms High Probability')
    plt.legend()
    # plt.savefig('nn2_hp_example_1.png')
    plt.show()
    plt.figure(6)
    plt.plot(high_prob_ex[4], 'tab:purple', label='Keyboard')  # Keyboard
    plt.plot(high_prob_ex[5], 'tab:brown', label='Mallet')  # Mallet
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms High Probability')
    plt.legend()
    # plt.savefig('nn2_hp_example_2.png')
    plt.show()
    plt.figure(7)
    plt.plot(high_prob_ex[9], 'tab:cyan', label='Vocal')  # Vocal
    plt.plot(high_prob_ex[7], 'tab:gray', label='Reed')  # Reed
    plt.plot(high_prob_ex[8], 'tab:olive', label='String')  # String
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms High Probability')
    plt.legend()
    # plt.savefig('nn2_hp_example_3.png')
    plt.show()
    plt.figure(8)
    plt.plot(high_prob_ex[6], 'tab:pink', label='Organ')  # Organ
    plt.plot(high_prob_ex[0], 'tab:blue', label='Bass')  # Bass
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms High Probability')
    plt.legend()
    # plt.savefig('nn2_hp_example_4.png')
    plt.show()

    mixed_prob = {}
    # Mixed probabilities
    # Loops through an finds the first example of each index and probability is around the same for other classes
    for x in range(10):
        index = 0
        flag = False
        for k, prediction in zip(range(len(test_y)), pred):
            if test_y[k] == x and flag is False:
                instr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                instr = np.delete(instr, x)
                for a in instr:
                    if (prediction[a] > prediction[test_y[k]] - 0.1) and (prediction[a] < prediction[test_y[k]] + 0.1):
                        ex = reshape[k].reshape(-1)
                        ex = ex[0:1600]
                        mixed_prob[x] = ex
                        flag = True
            index += 1

    plt.figure(9)
    plt.plot(mixed_prob[1], 'tab:orange', label='Brass')  # Brass
    plt.plot(mixed_prob[2], 'tab:green', label='Flute')  # Flute
    plt.plot(mixed_prob[3], 'tab:red', label='Guitar')  # Guitar
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms Mixed Probability')
    plt.legend()
    # plt.savefig('nn2_mixed_example_1.png')
    plt.show()
    plt.figure(10)
    plt.plot(mixed_prob[4], 'tab:purple', label='Keyboard')  # Keyboard
    plt.plot(mixed_prob[5], 'tab:brown', label='Mallet')  # Mallet
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms Mixed Probability')
    plt.legend()
    # plt.savefig('nn2_mixed_example_2.png')
    plt.show()
    plt.figure(11)
    plt.plot(mixed_prob[9], 'tab:cyan', label='Vocal')  # Vocal
    plt.plot(mixed_prob[7], 'tab:gray', label='Reed')  # Reed
    plt.plot(mixed_prob[8], 'tab:olive', label='String')  # String
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms Mixed Probability')
    plt.legend()
    # plt.savefig('nn2_mixed_example_3.png')
    plt.show()
    plt.figure(12)
    plt.plot(mixed_prob[6], 'tab:pink', label='Organ')  # Organ
    plt.plot(mixed_prob[0], 'tab:blue', label='Bass')  # Bass
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Reconstructed Waveforms Mixed Probability')
    plt.legend()
    # plt.savefig('nn2_mixed_example_4.png')
    plt.show()


'''
main conditional guard
The following condition checks whether we are running as a script.
If the file is being imported, don't run the test code.
'''
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage1: project2.py 1 model_index train_set_filename valid_set_filename")
        print("Usage2: project2.py 2 model_index test_set_filename model_filename")
        print("Please add the filenames of the datasets to be read and model index to "
              "system arguments (Edit configuration -> Parameters)")
        sys.exit(-1)

    # use any one of the GPUs available (just one)
    tf.config.set_soft_device_placement(True)

    task = int(sys.argv[1])

    # train a model, save it and its history, and draw learning curves
    if task == 1:
        # load training set
        print("start loading and processing training dataset")
        train, train_label = read_data(sys.argv[3], 289205, int(sys.argv[2]))
        print("finished loading training dataset")

        # load validation set
        print("start loading and processing validation dataset")
        valid, valid_label = read_data(sys.argv[4], 12678, int(sys.argv[2]))
        print("finished loading validation dataset")

        # train model
        model, model_history = train_model(train, train_label, valid, valid_label, int(sys.argv[2]))

        # save model and its history
        model.save("project2.model")
        np.save('history.dat', model_history.history)

        # draw learning curves
        make_curves(model_history.history)

    # evaluate a trained model
    elif task == 2:
        # load testing set
        print("start loading and processing testing dataset")
        test, test_label = read_data(sys.argv[3], 4096, int(sys.argv[2]))
        print("finished loading testing dataset")

        # load trained model
        model = tf.keras.models.load_model(sys.argv[4])
        model.summary()

        # draw learning curves, evaluate the model, and store the weights
        evaluate_model(model, test, test_label)

        # draw graphs
        if int(sys.argv[2]) == 1 or int(sys.argv[2]) == 2:
            make_plots(model, test, test_label)

    else:
        print("No such task. Please try again.")
        print("Usage1: project1.py 1 model_index train_set_filename valid_set_filename")
        print("Usage2: project1.py 2 model_index test_set_filename model_filename")

    # load data from already processed dataset
    '''
    train = np.load(sys.argv[], allow_pickle=True)
    train_label = np.load("I:/dataset/3/train_label3.dat", allow_pickle=True)
    valid = np.load("I:/dataset/3/valid3.dat", allow_pickle=True)
    valid_label = np.load("I:/dataset/3/valid_label3.dat", allow_pickle=True)
    test = np.load("I:/dataset/3/test3.dat", allow_pickle=True)
    test_label = np.load("I:/dataset/3/test_label3.dat", allow_pickle=True)
    model_history = np.load("I:/dataset/3/history.dat", allow_pickle=True)
    '''
