from __future__ import annotations
from typing import Callable, Dict, List, Union, Tuple

from copy import deepcopy
import random
import torch

from features.feature_extraction import extract_single_participant_features
import features.feature_shaping as feature_shaping

class InputOutput:
    """
    Class that stores input features with respective target outputs.
    """
    def __init__(self, input_data, target_output):
        self.input = input_data # Torch tensor
        self.output = target_output # torch tensor

    def __repr__(self):
        return f"insize {self.input.size()} & outsize {self.output.size()}"
    def combine(self, other: InputOutput) -> InputOutput:
        return InputOutput( # TODO: check axis of cat, 0 seems correct...
            torch.cat((self.input, other.input)),
            torch.cat((self.output, other.output))
        )

class TrainTest:
    """
    Class that stores training and testing data, and facilitates
    masking of the input and output data to drop columns.

    To retrieve the input and output training/testing data,
    optional parameters are as follows
        col_subset: True to indicate that a subset of columns
            should be taken
        cols_to_keep: A List of str, column names of the columns
            to keep
    """
    def __init__(
            self,
            train_input, train_output,  # Torch tensors
            test_input, test_output,    # Torch tensors
            col_names                   # List[str]
    ):
        self.train = InputOutput(train_input, train_output)
        self.test = InputOutput(test_input, test_output)
        self.col_names = col_names

    def _get_data(self, mode:str, col_subset:bool=False, cols_to_keep:List[str]=None):
        """
        Private helper to retrieve "train" or "test" <mode> data.
        """
        io = getattr(self, mode)
        if col_subset:
            to_keep_indices = [
                self.col_names.index(item) for item in cols_to_keep]
            return io.input[:, :, to_keep_indices], io.output[:, :, to_keep_indices]
        else:
            return io.input, io.output
    def train_data(self, col_subset=False, cols_to_keep=None):
        return self._get_data("train", col_subset, cols_to_keep)
    def test_data(self, col_subset=False, cols_to_keep=None):
        return self._get_data("test", col_subset, cols_to_keep)
    def combine(self, other: TrainTest) -> TrainTest:
        if self.col_names != other.col_names:
            raise RuntimeError("TrainTest objects are not compatible in one or more attributes.")
        train = self.train.combine(other.train)
        test = self.test.combine(other.test)
        return TrainTest(
            train.input, train.output,
            test.input, test.output,
            self.col_names
        )

    def __repr__(self):
        return f"{{Train: {self.train} Test: {self.test}\}}"

class DataSplit:
    """
    Represents train-test split for data, including support for
    cross-validation.

    Attributes
        <col_names>: a List of strings, the name of each feature in (column of) the input dataset, appearing in the
            same order as the columns of the data set.
        <input_data>: a torch Tensor of all the input data
        <target_outputs>: a torch Tensor of all the target outputs
        <train_fraction>: a float representing the fraction of data used for training. Default is 0.8, i.e. 80%
        <test_fraction>: a float representing the fraction of data used for testing. Default is 0.2, i.e. 20%
        <participants>: a List of the string names of each participant
        <participant_idx>: A List of integers; for row <i> of data in <input_data>,
            participant_idx[i] is the index of that participant in <participants>.

            For example, if <participants> is ["DEW001", "DEW008"] and <participants_index> is [0, 0, 0, 1, 1],
            then <input_data> has, in the following order, 3 rows of data corresponding to "DEW001" followed by
            2 rows of data corresponding to "DEW008".

            The number of unique values in <participant_idx> should be exactly equal to length of <participants>.
        <data_split>: Dict[str, List[TrainTest]]
    """
    def __init__(
            self,
            participants,
            participant_indices,
            input_data,
            target_outputs,
            train_fraction=0.8,
            col_names=None,
            cross_validate=False
    ):
        self.col_names = col_names
        self.input_data, self.target_outputs = input_data, target_outputs
        self.input_size = self.input_data.size()[2]

        self.train_fraction, self.test_fraction = train_fraction, 1 - train_fraction
        self.__num_data_points = self.input_data.size()[0]
        self.num_train = round(self.__num_data_points * self.train_fraction)
        self.num_test = self.__num_data_points - self.num_train

        self.participants, self.participant_idx = participants, participant_indices
        self.cross_validate = cross_validate

        print(f"Constructed data split object with input data size {self.input_data.size()}")
        self.data_split = {}
        if len(self.participants) == 1:
            self.data_split["batch"] = []
            self.__batch()
        else:
            self.__run_all_split()

    def __metadata_eq__(self: DataSplit, other: DataSplit):
        my_ds_keys = list(self.data_split.keys())
        other_ds_keys = list(other.data_split.keys())
        my_ds_keys.sort()
        other_ds_keys.sort()

        return  self.input_size == other.input_size and \
                self.train_fraction == other.train_fraction and \
                self.col_names == other.col_names and \
                self.cross_validate == other.cross_validate and \
                my_ds_keys == other_ds_keys
    def __combine__(self: DataSplit, other: DataSplit) -> DataSplit:
        if not self.__metadata_eq__(other):
            raise RuntimeError("DataSplit objects are not compatible in one or more attributes.")
        elif self.cross_validate:
            raise RuntimeError("Support for combining DataSplit objects when cross-validation is active is not yet implemented./")
        else:
            for split_mode in self.data_split:
                train_test = self.data_split[split_mode][0]
                other_train_test = other.data_split[split_mode][0]
                combined_train_test = train_test.combine(other_train_test)
                self.data_split[split_mode] = [combined_train_test]
            # NOTE: check numbers match up
            # assume no overlapping participants
            self.__num_data_points += other.__num_data_points
            self.num_train += other.num_train
            self.num_test += other.num_test

            self.input_data = torch.cat((self.input_data, other.input_data))
            self.target_outputs = torch.cat((self.target_outputs, other.target_outputs))

            original_participants_len = len(self.participants)
            self.participants.extend(other.participants)
            self.participant_idx.extend([el + original_participants_len for el in other.participant_idx])

    @staticmethod
    def merge_datasplits(ds_dict: Dict[str, DataSplit]):
        # assumes ds_tuple is non-empty
        ds_list = list(ds_dict.values())
        first_ds_copy = deepcopy(ds_list[0])
        for ds in ds_list[1:]:
            first_ds_copy.__combine__(ds)
        return first_ds_copy

    def __run_split(self, method):
        """
        method:
            "split_by_participant"
            "split_withing_participant"
        """
        getattr(self, method)()

    def __run_all_split(self):
        all_methods = dir(self)
        for method in all_methods:
            if method.startswith("split"):
                subname = method.split("split_")[-1]
                self.data_split[subname] = []
                self.__run_split(method)

    def __batch(self):
        """
        (train_fraction * 100) % of total data points used for training
        (test_fraction * 100) % of total data points used for testing
        """
        method_name = "batch"
        all_indices = [i for i in range(self.__num_data_points)]
        random.shuffle(all_indices)

        self.num_partitions = self.__num_data_points // self.num_test if self.cross_validate else 1
        for i in range(self.num_partitions):
            test_indices = all_indices[i * self.num_test : (i + 1) * self.num_test]
            train_indices =  all_indices[:i * self.num_test] + all_indices[(i + 1) * self.num_test:]
            self.data_split[method_name].append(
                TrainTest(
                    self.input_data[train_indices, :, :], self.target_outputs[train_indices],
                    self.input_data[test_indices, :, :], self.target_outputs[test_indices],
                    col_names=self.col_names
                )
            )

    def __get_participant_data_by_participant(self, desired_idx):
        i = self.participant_idx.index(desired_idx)
        next_i = i + self.participant_idx.count(desired_idx)
        return self.input_data[i:next_i, :, :], self.target_outputs[i:next_i, :, :]

    def split_by_participant(self):
        """
        i participants used for training
        n - i participants used for testing

        where i is the integer value obtained via
            round(num_participants * train_fraction)
        """
        method_name = "by_participant"
        num_participants = len(self.participants)
        num_participants_train = round(num_participants * self.train_fraction)
        num_participants_test = num_participants - num_participants_train
        all_indices = [i for i in range(num_participants)]
        random.shuffle(all_indices)

        num_partitions = num_participants // num_participants_test if self.cross_validate else 1
        for i in range(num_partitions):
            train_input, train_output = [], []
            test_input, test_output = [], []
            train_indices =  all_indices[:i * num_participants_test] + all_indices[(i + 1) * num_participants_test:]
            for j in all_indices:
                sub_input, sub_output = self.__get_participant_data_by_participant(j)
                if j in train_indices:
                    train_input.append(sub_input)
                    train_output.append(sub_output)
                else:
                    test_input.append(sub_input)
                    test_output.append(sub_output)
            self.data_split[method_name].append(
                TrainTest(
                    torch.cat(train_input), torch.cat(train_output),
                    torch.cat(test_input), torch.cat(test_output),
                    col_names=self.col_names
                )
            )

    def split_within_participant(self):
        method_name = "within_participant"
        participant_randi = []
        participant_start_end = []

        start_i = 0
        for i in range(len(self.participants)):
            num_entries = self.participant_idx.count(i)
            end_i = start_i + num_entries
            curr_indices = [j for j in range(start_i, end_i)]
            random.shuffle(curr_indices)
            participant_randi.append(curr_indices)
            participant_start_end.append((start_i, end_i, num_entries))
            start_i = end_i

        num_partitions = self.__num_data_points // self.num_test if self.cross_validate else 1
        for partition_i in range(num_partitions):
            train_input, train_output = [], []
            test_input, test_output = [], []
            for participant_i in range(len(self.participants)): # for each participant
                num_entries = participant_start_end[participant_i][-1]
                num_entries_test = round(num_entries * self.test_fraction)
                curr_indices = participant_randi[participant_i]
                train_indices = curr_indices[:partition_i * num_entries_test] + \
                                curr_indices[(partition_i + 1) * num_entries_test:]
                for idx in curr_indices:
                    if idx in train_indices:
                        train_input.append(self.input_data[idx:idx+1, :, :])
                        train_output.append(self.target_outputs[idx:idx+1, :, :])
                    else:
                        test_input.append(self.input_data[idx:idx+1, :, :])
                        test_output.append(self.target_outputs[idx:idx+1, :, :])

            self.data_split[method_name].append(
                TrainTest(
                    torch.cat(train_input), torch.cat(train_output),
                    torch.cat(test_input), torch.cat(test_output),
                    col_names=self.col_names
                )
            )
            
def train_model_one_epoch(
    input_features,  # batch, all, or single
    target_outputs,
    model,
    criterion,
    optimizer
):
    """

    One iteration on one data point for LSTM Training
    <input_features>:     a torch tensor with dimension <N> x <L> x <Hin>
    <target_outputs>:
    <model>:         the torch module that will be trained to fit the provided
                    <input_data> to match the <target_output>
    <criterion>:     a torch.nn loss function, see
                    https://pytorch.org/docs/stable/nn.html#loss-functions
    <optimizer>:    a torch.optim optimizer, see
                    https://pytorch.org/docs/stable/optim.html#algorithms
    <max_iters>:    an integer representing the number of iterations of training
                    to handle within the function; useful if all the data is
                    passed in at once. If a single data point is passed in to
                    the function call, <max_iters> should be 1 and the training
                    loop should be constructed to call the function every
                    iteration instead

    Returns loss values as per the <criterion> loss function after running a single epoch trained
    on all <N> points
    """
    optimizer.zero_grad()                           # reset the gradients to zero for each step to start fresh
    computed_y = model.forward(input_features)      # compute the output with the starting point of the model
    loss = criterion(computed_y, target_outputs)    # compute the error between the data points and the target output
    loss.backward()                                 # calculate the new gradients by backpropagating the MSE
    optimizer.step()                                # moves the weights in the direction of the gradients
    return loss



def format_features_for_training(
        full_data_dict, cols_to_analyze, participants_to_consider,
        max_window_size, sequence_length
):
    """
    Generates secondary features for all participants given in
    <participants_to_consider>, retaining only measures given in
    <cols_to_analyze>.

    Windowed secondary features are computed for window sizes from 1
    to <max_window_size> (inclusive).
    """
    # Pre-allocating list to store extracted features for each participant
    all_participants_features = [None] * len(participants_to_consider)
    all_participants_outputs = [None] * len(participants_to_consider)
    participants_indices = []

    print(f"Considering {len(participants_to_consider)} participants:"
          f" {participants_to_consider}")

    features_names = None
    num_participants = len(participants_to_consider)
    for i in range(num_participants):
        participant = participants_to_consider[i]
        print(f"Extracting secondary features for participant {participant}")
        metrics = extract_single_participant_features(full_data_dict[participant], cols_to_analyze,
                                                      max_window_size=max_window_size)
        processed_features = metrics.get_all_features()
        features_df = feature_shaping.extracted_features_to_dataframe(processed_features)
        features_tensor = feature_shaping.tensor_from_dataframe(features_df)  # rows: time, columns: features
        input_features, target_outputs = feature_shaping.generate_lstm_dataset(features_tensor, sequence_length)
        all_participants_features[i], \
            all_participants_outputs[i] = input_features, target_outputs

        if features_names is None:
            features_names = list(features_df.columns.values)

        num_entries_for_participant = input_features.shape[0]
        participants_indices.extend([i] * num_entries_for_participant)
    print(f"!! Completed extracting secondary features for all participants.")

    return torch.cat(all_participants_features), torch.cat(all_participants_outputs), \
           participants_indices, features_names



