import numpy as np
import tensorflow as tf
import h5py
from sklearn.metrics import classification_report, mean_squared_error as rmse
from ICFormer import ICFormer
import wandb
from wandb.keras import WandbCallback
import pandas as pd


def get_pred(model, time_steps, x_tests, y_test):
    """
    Prints predictions for test sets
    Parameters
    ----------
    model: h5py object, trained model
    x_test: list, test sets
    y_test: array, test set labels
    reshape: bool, if True, an extra dimension is added to the input
    DTW: bool, if True, the data is reshaped to the DTW model input shape
    """
    average = []

    for x, y in zip(x_tests, y_test):
        cycles = [200, 400, 600, 800, 1000]

        (
            predictions_reg,
            predictions_clf,
            attention_outputs,
            attention_weights,
            attention_outputs_sum,
        ) = model.predict(x)
        predictions_reg = predictions_reg.reshape(
            predictions_reg.shape[0], time_steps, predictions_reg.shape[1] // time_steps
        )

        predictions_LLI = np.zeros(len(cycles))
        predictions_LAMPE = np.zeros(len(cycles))
        predictions_LAMNE = np.zeros(len(cycles))

        for cycle in range(x.shape[1]):

            predictions_LLI[cycle] = rmse(
                y[:, cycle, 0], predictions_reg[:, cycle, 0], squared=False
            )
            predictions_LAMPE[cycle] = rmse(
                y[:, cycle, 1], predictions_reg[:, cycle, 1], squared=False
            )
            predictions_LAMNE[cycle] = rmse(
                y[:, cycle, 2], predictions_reg[:, cycle, 2], squared=False
            )

        df = pd.DataFrame(
            np.stack((predictions_LLI, predictions_LAMPE, predictions_LAMNE)),
            index=["LLI", "LAMPE", "LAMNE"],
            columns=[200, 400, 600, 800, 1000],
        )
        average.append(np.mean(df.mean(axis=1)))
        print(df)


def get_pred_clf(model, x_tests, y_test):

    for x, y in zip(x_tests, y_test):

        (
            predictions_reg,
            predictions_clf,
            attention_outputs,
            attention_weights,
            attention_outputs_sum,
        ) = model.predict(x)
        print(classification_report(y, np.round(predictions_clf)))


if __name__ == "__main__":

    # Pass your defaults to wandb.init
    run = wandb.init(project="ICFormer")
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    chemistry = "LFP"
    path = "./data/" + chemistry

    with h5py.File(path + "/datasets.h5", "r") as h5f:
        first_dataset = h5f["dataset_pre"]
        norm_values = first_dataset["norm_values"][()]
        min_val = norm_values[0]
        max_val = norm_values[1]

        dataset = h5f["dataset"]
        x_train = dataset["x_ICs_train"][()]
        y_regression_train = dataset["y_degradation_modes_train"][()]
        y_classification_train = dataset["y_silent_modes_train"][()].astype(np.float32)
    del first_dataset
    del dataset
    time_steps = x_train.shape[1]
    y_regression_train = y_regression_train.reshape(
        y_regression_train.shape[0],
        y_regression_train.shape[1] * y_regression_train.shape[2],
    )

    # load data
    with h5py.File(path + "/test/datasets.h5", "r") as h5f:
        test_datasets_ICs = h5f["x_test"][:]
        test_datasets_degradation_modes = h5f["y_test"][:]
        test_datasets_QLs = h5f["y_test_QLs"][:]
        test_datasets_silent_modes = h5f["y_test_silent_modes"][:].astype(np.float32)

    x_test = np.reshape(
        test_datasets_ICs,
        (
            test_datasets_ICs.shape[0] * test_datasets_ICs.shape[1],
            test_datasets_ICs.shape[2],
            test_datasets_ICs.shape[3],
        ),
    )
    y_regression_test = np.reshape(
        test_datasets_degradation_modes,
        (
            test_datasets_degradation_modes.shape[0],
            test_datasets_degradation_modes.shape[1],
            test_datasets_degradation_modes.shape[2]
            * test_datasets_degradation_modes.shape[3],
        ),
    )
    y_regression_test = np.reshape(
        y_regression_test,
        (
            y_regression_test.shape[0] * y_regression_test.shape[1],
            y_regression_test.shape[2],
        ),
    )
    y_classification_test = np.reshape(
        test_datasets_silent_modes,
        (test_datasets_silent_modes.shape[0] * test_datasets_silent_modes.shape[1],),
    )

    # ------------------------------ TRAIN ON BOTH TASKS ------------------------------

    # Instantiate the model
    model = ICFormer(
        look_back=x_train.shape[1],
        n_features=x_train.shape[2],
        num_transformer_blocks=4,
        num_heads=2,
        head_size=28,
        ff_dim=28,
        mlp_units=128,
        mlp_dropout=0.2,
        dropout=0.2,
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/ICFormer_best10_" + str(run.id) + ".hdf5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        WandbCallback(save_model=False),
    ]

    # Train the model
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))

    history = model.fit(
        x_train,
        {
            "y_regression": y_regression_train,
            "y_classification": y_classification_train,
        },
        validation_data=(
            x_test,
            {
                "y_regression": y_regression_test,
                "y_classification": y_classification_test,
            },
        ),
        epochs=1000,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate the model
    model.load_weights("./checkpoints/ICFormer_best_" + str(run.id) + ".hdf5")
    get_pred(model, time_steps, test_datasets_ICs, test_datasets_degradation_modes)
    get_pred_clf(model, test_datasets_ICs, test_datasets_silent_modes)
    # ----------------------------------------------------------------------------------
