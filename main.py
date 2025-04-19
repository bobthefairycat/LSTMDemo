# TODO: why are the losses so different? botched sample sizes?
import matplotlib.pyplot as plt
from lstm.model_training import DataSplit, train_model_one_epoch, format_features_for_training

from model import LSTMModel
from typing import List, Tuple, Dict
import torch
import pickle
import time
import os

MOOD_LABEL_TO_DESCRIPTION = {
    "mood1": "How happy versus sad do you feel right now? "
             "(1) Very cheerful/happy [...] (7) Very sad/depressed/unhappy",
    "mood2": "How much are you able to enjoy and feel pleasure in things? "
             "(1) Really enjoying things [...] (7) No pleasure or enjoyment",
    "mood3": "",
    "mood4": "",
    "mood5": "",
    "mood6": "",
    "mood7": "",
    "mood8": "",
    "mood9": "",
    "mood10": "",
    "mood11": ""
}

DATA_FNAME = "../data/pickled/2week_ema_data.pickle"
DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)

RESULTS_DIR = "testplots"

def filter_participants(all_data, participant_indices):
    all_participants = list(all_data.keys())  # Selecting subset of participants for analysis (default is all)
    participants_to_consider = all_participants if participant_indices is None \
        else list(filter(lambda x: x in participant_indices, all_participants))
    return participants_to_consider

def train_compare_raw_vs_augmented(
        all_data,
        sequence_length: int,
        hidden_state_dim_multiplier: float,
        num_layers: int,
        learning_rate: float,
        participant_index: List = None,
        max_window_size: int = 30,
        batch_size=10,
        num_epochs=50,
        split_mode="batch",
        split_data=None,
        cross_validate=False,
        counter=None
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if split_data is None:
        participants_to_consider = filter_participants(all_data, participant_index)
        # place the training data and target outputs into correctly-shaped tensors
        all_participants_features, \
            all_participants_outputs, \
            participant_indices, \
            features_names = format_features_for_training(
            all_data,  # dictionary pickled
            cols_to_analyze=list(MOOD_LABEL_TO_DESCRIPTION.keys()),
            participants_to_consider=participants_to_consider,
            max_window_size=max_window_size,
            sequence_length=sequence_length
        )

        all_participants_features = all_participants_features.to(DEVICE)
        all_participants_outputs = all_participants_outputs.to(DEVICE)
        ds = DataSplit(
            participants_to_consider,
            participant_indices,
            all_participants_features,
            all_participants_outputs,
            col_names=features_names,
            cross_validate=cross_validate
        )
    else:
        ds = split_data

    raw_features = list(filter(lambda x: x.startswith("raw"), ds.col_names))
    print(f"Now starting training on {DEVICE_NAME} for data split mode {split_mode}")
    print(f"Cross validation: {cross_validate}")

    cross_validation_limit = 1 if not cross_validate else len(ds.data_split[split_mode])
    xaxis_vals = [i for i in range(num_epochs)]
    fig, ax = plt.subplots(cross_validation_limit, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 9)
    fig.text(0.5, 0.95, f'Split {(" ").join(split_mode.split("_"))}', ha='center')
    fig.text(0.5, 0.04, 'Epoch', ha='center')
    fig.text(0.03, 0.5, 'L1 Loss', va='center', rotation='vertical')

    for cross_validation_idx in range(0, cross_validation_limit):
        i_train, _ = ds.data_split[split_mode][cross_validation_idx].train_data()
        i_test, _ = ds.data_split[split_mode][cross_validation_idx].test_data()
        print(f"\tDataset Train Size: {i_train.size()[0]}"
              f", Test Size: {i_test.size()[0]}")

        print(f"!!! {cross_validation_idx+1} of {cross_validation_limit} cross validation steps")
        lstm_aug = LSTMModel(
            ds.input_size,
            round(ds.input_size * hidden_state_dim_multiplier),
            num_layers,
            len(raw_features)  # predict only raw mood
        )
        lstm_raw = LSTMModel(
            len(raw_features),
            round(ds.input_size * hidden_state_dim_multiplier),
            num_layers,
            len(raw_features)
        )
        lstm_aug.to(DEVICE)  # send to GPU
        lstm_raw.to(DEVICE)

        criterion = torch.nn.L1Loss(reduction='sum')  # mean absolute error (MAE), summed
        optimizer_aug = torch.optim.Adam(
            lstm_aug.parameters(),
            lr=learning_rate
        )
        optimizer_raw = torch.optim.Adam(
            lstm_raw.parameters(),
            lr=learning_rate
        )

        min_loss_aug, min_loss_raw = 1000, 1000
        min_loss_aug_i, min_loss_raw_i = -1, -1

        loss_vals_aug, loss_vals_raw = [0] * num_epochs, [0] * num_epochs
        test_loss_vals_aug, test_loss_vals_raw = [0] * num_epochs, [0] * num_epochs

        train_set_input, _ = ds.data_split[split_mode][cross_validation_idx].train_data()
        test_set_input, _ = ds.data_split[split_mode][cross_validation_idx].test_data()
        raw_train_set_input, raw_train_set_output = ds.data_split[split_mode][cross_validation_idx].train_data(True, raw_features)
        raw_test_set_input, raw_test_set_output = ds.data_split[split_mode][cross_validation_idx].test_data(True, raw_features)

        for i in range(0, num_epochs):
            lstm_aug.train()
            lstm_raw.train()

            for batch_start_idx in range(0, ds.num_train, batch_size):
                end_idx = batch_start_idx + batch_size
                loss_aug = train_model_one_epoch(
                    train_set_input[batch_start_idx:end_idx, :, :],
                    raw_train_set_output[batch_start_idx:end_idx, :, :],
                    model=lstm_aug,
                    criterion=criterion,
                    optimizer=optimizer_aug
                )
                loss_raw = train_model_one_epoch(
                    raw_train_set_input[batch_start_idx:end_idx, :, :],
                    raw_train_set_output[batch_start_idx:end_idx, :, :],
                    model=lstm_raw,
                    criterion=criterion,
                    optimizer=optimizer_raw
                )
                loss_vals_aug[i] += loss_aug.cpu().detach().numpy()
                loss_vals_raw[i] += loss_raw.cpu().detach().numpy()

            loss_vals_aug[i] /= torch.numel(raw_train_set_output)
            loss_vals_raw[i] /= torch.numel(raw_train_set_output)

            lstm_aug.eval()
            lstm_raw.eval()
            test_prediction_aug = lstm_aug(test_set_input)
            test_prediction_raw = lstm_raw(raw_test_set_input)

            test_loss_vals_aug[i] = criterion(
                test_prediction_aug, raw_test_set_output).cpu().detach().numpy() / torch.numel(raw_test_set_output)
            test_loss_vals_raw[i] = criterion(
                test_prediction_raw, raw_test_set_output).cpu().detach().numpy() / torch.numel(raw_test_set_output)

            if test_loss_vals_aug[i] < min_loss_aug:
                min_loss_aug = test_loss_vals_aug[i]
                min_loss_aug_i = i

            if test_loss_vals_raw[i] < min_loss_raw:
                min_loss_raw = test_loss_vals_raw[i]
                min_loss_raw_i = i

            print(f"\rEpoch {i} of {num_epochs}", end="")

        print("\nTraining completed.")

        f = open(f"{RESULTS_DIR}/metadata_{counter}.txt", 'w')
        f.write(f"Augmented model\n")
        f.write(f"Size Train {i_train.size()[0]} Test {i_train.size()[0]}\n")
        f.write("Split Mode {split_mode}\nLearning Rate {learning_rate}\n")
        f.write(f"Cross validation {cross_validation_idx + 1} of {cross_validation_limit}\n")
        f.write(
            f"Num Layers {num_layers}\nSequence Length {sequence_length}\nHidden State Multiplier {hidden_state_dim_multiplier}\n")
        f.close()

        f = open(f"{RESULTS_DIR}/loss_vals_aug_{counter}.csv", 'w')
        f.write("Train, Test\n")
        for i in range(num_epochs):
            f.write(f"{loss_vals_aug[i]}, {test_loss_vals_aug[i]}\n")
        f.close()

        f = open(f"{RESULTS_DIR}/loss_vals_raw_{counter}.csv", 'w')
        f.write("Train, Test\n")
        for i in range(num_epochs):
            f.write(f"{loss_vals_raw[i]}, {test_loss_vals_raw[i]}\n")
        f.close()

        if cross_validation_limit != 1:
            ax_loss_aug, ax_loss_raw = ax[cross_validation_idx, 0], ax[cross_validation_idx, 1]
        else:
            ax_loss_aug, ax_loss_raw = ax[0], ax[1]

        helper_plot(ax_loss_aug, xaxis_vals, loss_vals_aug, test_loss_vals_aug, min_loss_aug_i, min_loss_aug)
        helper_plot(ax_loss_raw, xaxis_vals, loss_vals_raw, test_loss_vals_raw, min_loss_raw_i, min_loss_raw)

        if cross_validation_idx == 0:
            ax_loss_aug.set_title(f"Augmented model")
            ax_loss_raw.set_title(f"Raw model")
            ax_loss_aug.legend()
            ax_loss_raw.legend()
    plt.savefig(
        f"{RESULTS_DIR}/l1loss_{counter}.png"
    )
    plt.show()
    return ds

def helper_plot(ax, xaxis_vals, train_loss_vals, test_loss_vals, min_loss_i, min_loss):
    ax.plot(xaxis_vals, train_loss_vals, 'k', linewidth=3, label="Training set")
    ax.plot(xaxis_vals, test_loss_vals, 'r--', linewidth=3, label="Test set", )
    ax.vlines(x=min_loss_i, color='0.2', alpha=0.4, linestyle="--", ymin=0, ymax=min_loss)
    ax.set_ylim(0, 0.3)
    ax.grid(True)


if __name__ == "__main__":
    f = open(DATA_FNAME, 'rb')
    data = pickle.load(f) # NOTE: assumes data_loader.py has been run at least once
    f.close()

    plt.rcParams["font.size"] = "20"

    participants = list(data.keys())
    ds = {}
    counter = 0
    for participant in participants[0:3]:
        ds_tmp = train_compare_raw_vs_augmented(
            data,
            sequence_length=2,
            hidden_state_dim_multiplier=1,
            num_layers=2,
            learning_rate=0.01,
            batch_size=50,
            num_epochs=2,
            split_mode="batch",
            split_data=None,
            cross_validate=False,
            counter=counter,
            participant_index=[participant]
        )
        counter += 1
        ds[participant] = ds_tmp

    print("merged")
    combined_ds = DataSplit.merge_datasplits(ds)
    for mode, split in combined_ds.data_split.items():
        print(f"{mode}: {split}")

    print("original")
    for k, v in ds.items():
        print(f"{k}: {v.data_split}")