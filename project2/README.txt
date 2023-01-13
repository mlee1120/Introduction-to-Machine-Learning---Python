Usage1: project2.py 1 model_index train_set_filename valid_set_filename

Usage1 is for training models:
model_index can be 1, 2, or 3 for training our LSTM, BLSTM, and the bonus models respectively.
train_set_filename is the file path name of the training dataset.
valid_set_filename is the file path name of the validation dataset.
It takes about 10 minutes to process the dataset (10 seconds for testing set only).
(We serialized the processed datasets, so we did not have to wait 10 minutes everytime we trained a model)
1 1 nsynth-train.tfrecord nsynth-valid.tfrecord
1 2 nsynth-train.tfrecord nsynth-valid.tfrecord
1 3 nsynth-train.tfrecord nsynth-valid.tfrecord

Usage2: project2.py 2 model_index test_set_filename model_filename

Usage2 is for evaluating trained models and drawing plots for project 2.
odel_index can be 1, 2, or 3 for our LSTM, BLSTM, and the bonus models respectively.
test_set_filename is the file path name of the testing dataset.
model_filename is the file path name of a tranined model
2 1 nsynth-test.tfrecord LSTM.model
2 2 nsynth-test.tfrecord BLSTM.model
2 3 nsynth-test.tfrecord Bonus.model
(use these three system arguments for evaluating our models)