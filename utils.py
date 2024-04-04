import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as pchip
import h5py
import scipy.io as sio
from sklearn.metrics import (
    classification_report,
    mean_squared_error as rmse,
    pairwise_distances,
    r2_score,
)
from scipy.optimize import curve_fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import display
import pandas as pd

UI_STEP = 0.0005
MIN_V_LFP = 3.20
MAX_V_LFP = 3.50
MIN_V_NCA = 3.65
MAX_V_NCA = 4.20
MIN_V_NMC = 3.44
MAX_V_NMC = 4.28
CURVE_SIZE = 128

LFP_MIN = 0.0031566024955984595
LFP_MAX = 2.736867845392978
NCA_MIN = 0
NCA_MAX = 0.2682065162447511
NMC_MIN = 0
NMC_MAX = 0.1914037352896408


# --------------------------------------------------READ DATA--------------------------------------------------
def read_mat(file_name):
    """
    Reads a .mat file and returns the data as a numpy array

    Parameters
        ----------
        file_name: str, path to the .mat file
    """

    return sio.loadmat(file_name)


def read_mat_hdf5(file_name, field_name):
    """
        Opens mat file as a numpy array with hdf5
    Must retrieve all the indexes: advanced indexing in h5py is not nearly as general as with np.ndarray,
    an exception will be raised if the indexes are not continuous

    Parameters
        ----------
        file_name: str, path to the .mat file
    field_name: str, name of the field inside the .mat file
    """

    with h5py.File(file_name, "r") as f:
        data = f[field_name][:]
    return data


# -------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------PROCESS DATA--------------------------------------------------
def get_minmaxV(material):
    """
    Returns the range voltage in which to study the IC curves
    Parameters
    ----------
    material: numpy array, chemistry to study
    Returns
    -------
    min_v, max_v, path: numpy arrays, min and max voltage values and path where data is located
    """

    min_v = -1
    max_v = -1
    path = ""
    if material == "LFP":
        path = "./mat/LFP"
        min_v = MIN_V_LFP
        max_v = MAX_V_LFP
    elif material == "NCA":
        path = "./mat/NCA"
        min_v = MIN_V_NCA
        max_v = MAX_V_NCA
    elif material == "NMC":
        path = "./mat/NMC"
        min_v = MIN_V_NMC
        max_v = MAX_V_NMC
    else:
        print("ERROR: Chemistry not found")
        return -1
    if min_v == -1 or max_v == -1 or path == "":
        print("ERROR: Chemistry not found")
        return -1
    return min_v, max_v, path


def IC(u, q, ui_step=0.0005, minV=3.2, maxV=3.5):
    """
    Get the ICA data for a given voltage curve

    Parameters
    ----------
    u: numpy array, voltage curve
    q: numpy array, capacity curve
    ui_step: float, step of interpolation
    minV: float, minimum voltage of the IC curve
    maxV: float, maximum voltage of the IC curve

    Returns
    -------
    ui, dqi: numpy arrays, interpolated voltage and derivative of capacity
    """

    # voltages values for which capacity is interpolated
    ui = np.arange(minV, maxV, ui_step)
    qi = np.interp(ui, u, q)
    return ui[1:], np.diff(qi)


def reduce_size(ui, dqi, size):
    """
    Reduces the length of the IC data to a given size

    Parameters
    ----------
    ui: numpy array, voltage curve
    dqi: numpy array, derivative of capacity (IC)
    size: int, size at which to reduce the IC data

    Returns
    -------
    numpy array, reduced IC
    """

    curve = pchip(ui, dqi)
    ui_reduced = np.linspace(min(ui), max(ui), size)
    return curve(ui_reduced)


def retrieve_curves(chemistry, indices, V, Q):
    """
    Calculates the IC curves for a set of samples.

    Parameters
    ----------
    indices : numpy array
        An array with the indices of the voltage curves associated with each sample and cycle.
        The shape of the array is (num_samples, num_cycles).
    V : numpy array
        An array with the voltage curves. The shape of the array is (num_voltage_curves, len_voltage_curves).
    Q : numpy array
        An array with the capacity curves. The shape of the array is (len_capacity_curves).
    len_curves : int
        The number of points to reduce the IC curves to.

    Returns
    -------
    IC_curves : numpy array
        An array with the IC curves for each sample and cycle. The shape of the array is
        (num_samples, num_cycles, len_curves).
    """

    num_samples, num_cycles = indices.shape
    IC_curves = np.zeros((num_samples, num_cycles, CURVE_SIZE))
    for duty in range(num_samples):
        for cycle in range(num_cycles):
            # indices[duty, cycles] contains the index of the voltage curve associated with this cycle
            index_voltage_curve = indices[duty, cycle]
            # if the index is negative, it means that there is no voltage curve associated with this cycle
            if index_voltage_curve >= 0:
                # calculate IC curve
                if chemistry == "LFP":
                    # if the voltage curve is above the minimum voltage then the IC curve cannot be calculated, so skip it
                    if (
                        np.any(V[index_voltage_curve] < MIN_V_LFP) == False
                        or np.any(V[index_voltage_curve] > MAX_V_LFP) == False
                    ):
                        break
                    ui, dqi = IC(
                        V[index_voltage_curve, :], Q, minV=MIN_V_LFP, maxV=MAX_V_LFP
                    )
                elif chemistry == "NCA":
                    if (
                        np.any(V[index_voltage_curve] < MIN_V_NCA) == False
                        or np.any(V[index_voltage_curve] > MAX_V_NCA) == False
                    ):
                        break
                    ui, dqi = IC(
                        V[index_voltage_curve, :], Q, minV=MIN_V_NCA, maxV=MAX_V_NCA
                    )
                elif chemistry == "NMC":
                    if (
                        np.any(V[index_voltage_curve] < MIN_V_NMC) == False
                        or np.any(V[index_voltage_curve] > MAX_V_NMC) == False
                    ):
                        break
                    ui, dqi = IC(
                        V[index_voltage_curve, :], Q, minV=MIN_V_NMC, maxV=MAX_V_NMC
                    )
                # reduce to len_curves points
                IC_curve = reduce_size(ui, dqi, CURVE_SIZE)
                IC_curves[duty, cycle, :] = IC_curve
    return IC_curves


def interpolate_data(IC_curves, QL, current_cycles, query_cycles):
    """
    Interpolate the IC_curves for a query_cycles array using pchip interpolation.

    Parameters
    ----------
    IC_curves : ndarray
        A 3D NumPy array of shape (sample, current_cycles, IC_curve) where each sample represents the evolutions of the IC curves of a duty cycle over a set of cycles.
    QL: ndarray
        A 2D NumPy array of shape (sample, current_cycles) where each sample represents the evolutions of the QL curves of a duty cycle over a set of cycles.
    query_cycles : ndarray
        A 1D NumPy array of cycle-values at which the interpolated curves should be evaluated.

    Returns
    -------
    interpolated_ICs : ndarray
        A 3D NumPy array of shape (sample, query_cycles, IC_curve).
    """
    interpolated_ICs = []
    interpolated_QL = []
    assert IC_curves.shape[0] == QL.shape[0]
    for sample in range(IC_curves.shape[0]):
        # interpolate QL
        curve = pchip(current_cycles, QL[sample])
        interpolated_QL.append(curve(query_cycles))
        # interpolate IC curves
        interpolated_sample = []
        for point_evolution in IC_curves[sample].T:
            curve = pchip(current_cycles, point_evolution)
            interpolated_sample.append(curve(query_cycles))
        interpolated_ICs.append(np.array(interpolated_sample).T)
    return np.array(interpolated_ICs), np.array(interpolated_QL)


def get_increasing_samples(QL):
    """
    Returns the samples with increasing capacity loss
    """
    # Initialize a boolean array to store whether each sample is non-increasing or not
    increasing = np.zeros(QL.shape[0], dtype=bool)
    # Iterate over each sample
    for i, sample in enumerate(QL):
        # Check if the capacity loss is increasing
        increasing[i] = (sample[1:] >= sample[:-1]).all()
    # Return the samples with increasing capacity loss
    return np.where(increasing)[0]


def find_most_similar(ICs, context):
    # Get the number of samples from the first dimension of the ICs array
    num_samples = ICs.shape[0]

    # Extract the IC curves for each sample up to cycle context
    ic_curves = ICs[:, :context, :]

    # Flatten the IC curves to a 2D array
    ic_curves = ic_curves.reshape(num_samples, -1)

    # Calculate the pairwise distances between all IC samples
    distances_IC = pairwise_distances(ic_curves, metric="cosine")

    # For each sample, find the indices of the 5 most similar samples
    # that are not the sample itself
    # return [np.argsort(d)[1:6] for d in distances], distances
    return distances_IC


def filter_similar_samples(distances_IC, QLs):
    samples_to_remove = []
    for current_sample_index, current_sample_distances in enumerate(distances_IC):
        for compared_sample_index, dist in enumerate(current_sample_distances):
            # if the distance is less than 0.001 it means that the samples have the same IC curves
            # dist!=0 is to avoid removing the sample itself
            if dist < 0.01 and dist != 0:
                # if the distance of QL is more than 0.002 it means that despite having the same IC curves they evolve differently
                # then append the one that has more degradation because that is an unpredictable event
                # if distances_QL[current_sample_index, context:][compared_sample_index] > 0.002 and QLs[compared_sample_index, context:][-1] > QLs[current_sample_index, context:][-1]:
                # if there is more than 10% difference in QL then remove the sample
                # print("Current sample:", current_sample_index, "Compared sample:", compared_sample_index, "Distance:", dist, "QL difference:", QLs[compared_sample_index][-1] - QLs[current_sample_index][-1])
                if (
                    QLs[compared_sample_index][-1] - QLs[current_sample_index][-1]
                    > 0.10
                ):
                    samples_to_remove.append(compared_sample_index)
        # print(samples_to_remove)
        # break

    return np.unique(samples_to_remove)


def exponential(x, a, b):
    return a * np.exp(b * x)


def is_non_linear(curve):
    # Fit an exponential curve to the data
    params, params_covariance = curve_fit(
        exponential, np.arange(curve.shape[0]), curve, p0=[1, 1]
    )
    # Calculate the R-squared value of the fit
    r_squared = r2_score(curve, exponential(np.arange(curve.shape[0]), *params))
    # If the fit is good (R-squared value is greater than 0.9) add the curve to the list
    return r_squared > 0.98


def is_linear(y, tolerance=2):
    """
    Check whether the given curve defined by x and y follows a linear trend.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates.
        y (numpy.ndarray): 1D array of y-coordinates.
        tolerance (float): Tolerance for linear trend detection. Default is 0.01.

    Returns:
        bool: True if the curve follows a linear trend, False otherwise.
    """
    # Perform linear regression to get the slope and intercept
    x = np.arange(len(y)) + 1
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate the residuals
    residuals = y - (m * x + c)

    # Check if the residuals are within the tolerance
    return np.max(np.abs(residuals)) < tolerance


def is_superlinear(curve):
    return not is_linear(curve)


def find_linear_breakpoints(y_vals, threshold=0.05):
    """
    Finds the breakpoints in a linear curve represented by a sequence of y-values.
    Returns a list of indices where the curve stops being linear.
    """
    n = len(y_vals)
    if n <= 2:
        return []

    # Calculate slopes between consecutive points
    slopes = [(y_vals[i + 1] - y_vals[i]) for i in range(n - 1)]

    # Find the maximum deviation from the initial slope
    max_deviation = 0.0
    deviation_index = 0
    for i in range(1, n - 2):
        deviation = abs((slopes[i] - slopes[0]) / slopes[0])
        if deviation > max_deviation:
            max_deviation = deviation
            deviation_index = i

    # If the deviation exceeds the threshold, return the breakpoint
    if max_deviation > threshold:
        return [deviation_index + 1]
    else:
        return []


def label_knees(QL, degradation_modes):
    knee_labels = np.zeros((QL.shape[0], QL.shape[1]))

    for sample in range(QL.shape[0]):
        # calculate cutoff -> the first cycle in which the QL is 80%
        cutoff = np.where(QL[sample] == 80)[0]
        if len(cutoff) > 0:
            cutoff = cutoff[0]
        else:
            cutoff = QL[sample].shape[0]
        # it is checked if the QL is superlinear
        if is_superlinear(QL[sample, :cutoff]):
            flag_knee = False
            # knee_labes_sample = [flag_knee] # the first cycle is always labeled as not a knee
            # label every cycle
            for cycle_num in range(1, cutoff):
                if not flag_knee:
                    # take the QL up to the current cycle
                    QL_up_to_cycle = QL[sample, :cycle_num]
                    # take the degradation modes up to the current cycle
                    degradation_modes_up_to_cycle = degradation_modes[
                        sample, :cycle_num
                    ]
                    # if the main degradation mode is LLI and there is knee check when it occurs
                    if np.argmax(
                        degradation_modes_up_to_cycle[-1, :]
                    ) == 0 and is_superlinear(QL_up_to_cycle):
                        cycle_knee = find_linear_breakpoints(
                            degradation_modes_up_to_cycle[:, 0]
                        )
                        if len(cycle_knee) > 0:
                            # el cycle_knee debería coincidir con el ciclo actual COMPROBAR
                            flag_knee = True
                            print(
                                "There is a knee because LLI is exponential at cycle: ",
                                cycle_knee[0],
                            )
                        break
                    else:  # if the degradation mode is not LLI -> LAM
                        # check if the knee occurs in the next 600 cycles
                        # 3 includes 3*200 = 600 cycles
                        if (
                            cycle_num + 3 < cutoff
                        ):  # if there are at least 600 cycles after the current cycle
                            future_QL = QL[sample, : cycle_num + 3]
                            # plt.plot(100-future_QL)
                            # plt.ylim(0, 100)
                            # plt.show()
                            if is_superlinear(future_QL):
                                flag_knee = True
                knee_labels[sample][cycle_num] = 1 if flag_knee else 0
        # else:
        #    knee_labels.append([False] * QL.shape[1])
    return knee_labels


def gen_data(chemistry, window_size_cycle):
    """
    Preprocesses data and saves it to disk.
    It involves the following steps:
    1. Load data
    2. Calculate IC curves and capacity losses
    3. Label knees
    4. Apply sliding window over the data
    5. Remove duplicates
    6. Save data to disk

    Parameters
    ----------
    context_cycles : int
        The cycle number up to which the data is used as a context for
        predicting the capacity loss in the next horizon_cycles.
    horizon_cycles : int
        The cycle number up to which the models predict capacity loss.
    chemistry : str
        The chemistry of the battery samples.
        Possible values are 'LFP', 'NMC', 'NCA'.
    """
    # 1. ------------------- LOAD DATA -------------------
    cyc = [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        400,
        600,
        800,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
        2200,
        2400,
        2600,
        2800,
        3000,
    ]  # cycles for which there is data
    # as the cycles are not equally spaced, we need to select the cycles we want to use
    selected_cycles = [
        0,
        200,
        400,
        600,
        800,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
        2200,
        2400,
        2600,
        2800,
        3000,
    ]
    # the index of the selected cycles are looked up in the original array
    selected_cycles_index = np.array(
        [np.where(np.array(cyc) == el)[0][0] for el in selected_cycles]
    )

    # --------------------- Indices to get data from diagnosis dataset --------------------
    indices = read_mat("./data/" + chemistry + "/Vi.mat")["Vi"][
        :, selected_cycles_index
    ]  # indices where to look for the voltage curve in the diagnosis dataset
    indices = indices - 1  # in matlab, indices start at 1, in python they start at 0
    indices = indices.astype(
        np.int32
    )  # change dtype of indices to int -> now nans are negative numbers

    print("Total samples: ", indices.shape[0])
    # assert indices.shape[0] == QL.shape[0]

    # ----------------- Degradation modes and Capacity loss ---------------
    degradations = read_mat("./data/" + chemistry + "/path_prognosis.mat")[
        "path_prognosis"
    ].T[:, selected_cycles_index]
    degradation_modes = degradations[:, :, 0:3].astype(np.float32)
    QL = degradations[:, :, 3].astype(np.float32)
    QL[QL == 0] = 80  # needed in order to check if the QL is increasing
    QL[:, 0] = 0  # there is not any capacity loss in the first cycle
    assert indices.shape[0] == QL.shape[0] == degradation_modes.shape[0]

    # --------------- Voltage and Capacity --------------
    V = read_mat_hdf5("./data/" + chemistry + "/volt.mat", "volt")
    Q = read_mat("./data/Q.mat")["Qnorm"].flatten()

    # 2. ------------------- REMOVE DUPLICATES ------------------- -> for some reason the dataset contains duplicates
    _, unique_indices = np.unique(indices, return_index=True, axis=0)
    indices = indices[unique_indices]
    QL = QL[unique_indices]
    degradation_modes = degradation_modes[unique_indices]
    print("Total samples after removing duplicates: ", indices.shape[0])
    assert indices.shape[0] == QL.shape[0] == degradation_modes.shape[0]

    # 3. ------------------- REMOVE SAMPLES THAT GAIN CAPACITY ------------------- -> for some reason the dataset contains samples that gain capacity
    valid_indices = get_increasing_samples(QL)
    indices = indices[valid_indices]
    QL = QL[valid_indices]
    degradation_modes = degradation_modes[valid_indices]
    print("Total samples after removing samples that gain capacity: ", indices.shape[0])

    # 4. ------------------- CALCULATE IC CURVES -------------------
    IC_curves = retrieve_curves(chemistry, indices, V, Q)

    # Calculate the minimum and maximum values
    min_val = np.min(IC_curves)
    max_val = np.max(IC_curves)
    # Normalize using the minimum and maximum values
    IC_curves = (IC_curves - min_val) / (max_val - min_val)

    assert (
        indices.shape[0]
        == QL.shape[0]
        == degradation_modes.shape[0]
        == IC_curves.shape[0]
    )

    # 5. ------------------- LABEL KNEES -------------------
    knee_labels = label_knees(QL, degradation_modes)

    # 6. ------------------- SLINDING WINDOW AND GET PAIRS X,Y -------------------
    window_size = selected_cycles.index(window_size_cycle) + 1

    x_ICs = []
    y_degradation_modes = []
    y_QLs = []
    y_knees = []
    sample_index = []

    num_samples, num_cycles, series_len = IC_curves.shape

    for sample_num in range(num_samples):
        for cycle_num in range(num_cycles - window_size + 1):
            ICs_cur = IC_curves[sample_num, cycle_num : cycle_num + window_size, :]
            degradation_modes_cur = degradation_modes[
                sample_num, cycle_num : cycle_num + window_size, :
            ]
            QLs_cur = QL[sample_num, cycle_num : cycle_num + window_size]
            knee_cur = knee_labels[sample_num][cycle_num : cycle_num + window_size]
            # if any of the degradation modes is nan or degradation is 0.8 or there is no IC curve skip the sample
            if (
                np.any(np.isnan(degradation_modes_cur))
                or np.any(QLs_cur == 80)
                or np.any(ICs_cur == 0.0)
            ):
                break
            else:
                x_ICs.append(ICs_cur)
                y_degradation_modes.append(degradation_modes_cur)
                y_QLs.append(QLs_cur)
                y_knees.append(knee_cur)
                sample_index.append(sample_num)

    x_ICs = np.array(x_ICs)
    y_degradation_modes = np.array(y_degradation_modes)
    y_QLs = np.array(y_QLs)
    y_knees = np.array(y_knees)
    sample_index = np.array(sample_index)
    assert (
        x_ICs.shape[0]
        == y_degradation_modes.shape[0]
        == y_QLs.shape[0]
        == y_knees.shape[0]
        == sample_index.shape[0]
    )

    # 7. ------------------- REMOVE SAMPLES IN WHICH THE IC CURVES ARE THE SAME -------------------
    _, unique_indices = np.unique(x_ICs, return_index=True, axis=0)
    # update the arrays
    x_ICs = x_ICs[unique_indices]
    y_degradation_modes = y_degradation_modes[unique_indices]
    y_QLs = y_QLs[unique_indices]
    y_knees = y_knees[unique_indices]
    sample_index = sample_index[unique_indices]
    print(x_ICs.shape)
    print(y_degradation_modes.shape)
    print(y_QLs.shape)
    print(y_knees.shape)
    print(sample_index.shape)

    # 8. ------------------- SAVE THE DATA -------------------
    with h5py.File("./data/" + chemistry + "/datasets_prueba.h5", "w") as h5f:
        # Create a group to store the dataset
        first_dataset = h5f.create_group("dataset")
        # Save the IC_curves, QLs and degradation_modes arrays to the group
        first_dataset.create_dataset("IC_curves", data=IC_curves)
        first_dataset.create_dataset("QLs", data=QL)
        first_dataset.create_dataset("degradation_modes", data=degradation_modes)
        first_dataset.create_dataset("knee_labels", data=knee_labels)
        first_dataset.create_dataset("norm_values", data=[min_val, max_val])

        # Create a group to store the dataset slices
        second_dataset = h5f.create_group("dataset_slices")
        # Save the x_ICs_train, x_ICs_test, x_ICs_val, y_degradation_modes_train, y_degradation_modes_test, y_degradation_modes_val, y_QLs_train, y_QLs_test, y_QLs_val, y_silent_modes_train, y_silent_modes_test, y_silent_modes_val, sample_index_train, sample_index_test, sample_index_val arrays to the group
        second_dataset.create_dataset("x_ICs", data=x_ICs)
        second_dataset.create_dataset("y_degradation_modes", data=y_degradation_modes)
        second_dataset.create_dataset("y_QLs", data=y_QLs)
        second_dataset.create_dataset("y_knees", data=y_knees)
        second_dataset.create_dataset("sample_index", data=sample_index)


def normalise_data(data, min_val, max_val, low=0, high=1):
    """
    Normalises the data to the range [low, high]

    Parameters
    ----------
    data: numpy array, data to normalise
    min: float, minimum value of data
    max: float, maximum value of data
    low: float, minimum value of the range
    high: float, maximum value of the range

    Returns
    -------
    normalised_data: float, normalised data
    """
    normalised_data = (data - min_val) / (max_val - min_val)
    normalised_data = (high - low) * normalised_data + low
    return normalised_data


def save_training_data(chemistry, context_cycles=800):

    print("Generating data for chemistry: {}".format(chemistry))
    gen_data(chemistry, context_cycles)


# -----------------------------------------------TRAIN AND INFERENCE-------------------------------------------
def convert_to_input_data(ui_new, Q, size, material):
    """
    Converts the voltage values of the real cells to the input data for the neural network
    Parameters
    ----------
    ui_new: array, voltage values of the cell at each cycle in percentage
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    size: int, the length of the curves
    material: str, chemistry of the cell
    Returns
    -------
    x_test: array, the input data for the neural network
    """
    min_v, max_v, path = get_minmaxV(material)
    samples = []
    for sample in range(len(ui_new)):
        # convert to IC
        ui_sample, dqi_sample = IC(ui_new[sample], Q, UI_STEP, min_v, max_v)
        # reduce size
        new_sample = reduce_size(ui_sample, dqi_sample, size)
        samples.append(new_sample)
    x_test = np.array(samples)
    return x_test


def train_model(
    model,
    x_train,
    y_regression_train,
    y_classification_train,
    x_val,
    y_regression_val,
    y_classification_val,
    batch_size,
    name,
):

    # compile the model
    model.compile(
        optimizer="adam",
        loss={"y_regression": "mse", "y_classification": "binary_crossentropy"},
        metrics={"y_regression": ["mae"], "y_classification": ["accuracy"]},
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", mode="min", patience=15, restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="../saved/" + name + ".h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        ),
    ]

    # train the model
    history = model.fit(
        x_train,
        {
            "y_regression": y_regression_train,
            "y_classification": y_classification_train,
        },
        validation_data=(
            x_val,
            {
                "y_regression": y_regression_val,
                "y_classification": y_classification_val,
            },
        ),
        epochs=1000,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Train the model
    return model


def results_report(
    model, x_test, y_regression_test, y_classification_test, reshape=False
):
    # evaluate the model
    predictions_reg, predictions_clf = model.predict(x_test)

    # regression
    if reshape:
        time_steps, degradation_modes = y_regression_test.shape[1:]
        predictions_reg = predictions_reg.reshape(
            predictions_reg.shape[0], time_steps, degradation_modes
        )

    print(
        "RMSE LLI: ",
        rmse(y_regression_test[:, :, 0], predictions_reg[:, :, 0], squared=False),
    )
    print(
        "RMSE LAMPE: ",
        rmse(y_regression_test[:, :, 1], predictions_reg[:, :, 1], squared=False),
    )
    print(
        "RMSE LAMNE: ",
        rmse(y_regression_test[:, :, 2], predictions_reg[:, :, 2], squared=False),
    )

    # classification
    predictions_clf = predictions_clf.reshape(-1)
    predictions_clf = np.round(predictions_clf)

    # Generate the report
    print(classification_report(y_classification_test, predictions_clf))


def plot_sample(sample, cycles, context):
    plt.figure(figsize=(30, 5))
    plt.title("IC curves evolution")
    X = np.linspace(0, cycles[0:context], sample.reshape(-1).shape[0])
    Y = sample.reshape(-1)
    plt.plot(X, Y, color="black")
    plt.margins(x=0, tight=True)
    # add a vertical line to indicate the horizon
    plt.axvline(x=cycles[context], color="r")
    plt.fill_between(
        X,
        np.ones(len(X)) * np.max(Y),
        0,
        where=np.less_equal(X, cycles[context]),
        color="lightblue",
    )
    plt.xlabel("Cycle #")
    plt.ylabel("Incremental Capacity (Ah/V)")
    plt.show()


def plot_prediction_distribution(y_true, y_pred):
    plt.figure(figsize=(15, 4))
    plt.hist(y_true, bins=100, color="green", alpha=0.5, label="true")
    plt.hist(y_pred, bins=100, color="orange", alpha=0.5, label="pred")
    plt.legend(loc="upper right")
    plt.title("value distribution")
    plt.show()


def plot_random_prediction(y_true, y_pred):
    sample = np.random.randint(0, y_true.shape[0])
    # plot the prediction for a random sample
    plt.figure(figsize=(15, 4))
    plt.plot(100 - y_true[sample], color="green", label="true")
    # plot the prediction with dotted line
    plt.plot(100 - y_pred[sample], color="orange", label="pred", linestyle="dashed")
    plt.title("prediction for sample {}".format(sample))
    plt.xlabel("cycle")
    plt.ylabel("Capacity")
    plt.legend(loc="upper right")
    plt.show()


def plot_prediction(cycles, y_true, y_pred, sample, context, horizon):
    # plot the prediction for a random sample
    plt.figure(figsize=(15, 4))
    plt.title("prediction for sample {}".format(sample))
    plt.plot(
        cycles[0 : context + 1], 100 - y_true[sample][0 : context + 1], color="green"
    )
    plt.plot(
        cycles[context : context + horizon],
        100 - y_true[sample][context : context + horizon],
        color="green",
        label="true",
    )
    # plot the prediction with dotted line
    plt.plot(
        cycles[context : context + horizon],
        100 - y_pred[sample],
        color="orange",
        label="pred",
        linestyle="dashed",
    )
    plt.axvline(x=cycles[context], color="r")
    plt.xlabel("cycle")
    plt.ylabel("Capacity")
    plt.legend(loc="upper right")
    # plt.show()
    return plt.gcf()


def plot_training_history(history):
    plt.figure(figsize=(15, 4))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["mape"], label="mape")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc="upper right")
    plt.show()


def get_ICs(curves, Q, minV, maxV):

    num_samples, num_cycles, _ = curves.shape
    IC_curves = np.zeros((num_samples, num_cycles, 128))
    for duty in range(num_samples):
        for cycle in range(num_cycles):
            ui, dqi = IC(curves[duty, cycle], Q, minV=minV, maxV=maxV)
            IC_curve = reduce_size(ui, dqi, 128)
            IC_curves[duty, cycle, :] = IC_curve
    return IC_curves


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
        display(df)


def evaluate_other_cells(chemistry, path, min_val, max_val, model, time_steps):
    if chemistry == "LFP":
        minV = MIN_V_LFP
        maxV = MAX_V_LFP
    elif chemistry == "NCA":
        minV = MIN_V_NCA
        maxV = MAX_V_NCA
    elif chemistry == "NMC":
        minV = MIN_V_NMC
        maxV = MAX_V_NMC

    cycles = [10, 50, 100, 200, 400, 1000]

    x_test_0 = read_mat(path + "/x_test_0.mat")["x_test"].T
    x_test_1 = read_mat(path + "/x_test_1.mat")["x_test"].T
    x_test_2 = read_mat(path + "/x_test_2.mat")["x_test"].T
    x_test_3 = read_mat(path + "/x_test_3.mat")["x_test"].T
    y_test = read_mat(path + "/y_test.mat")["y_test"][:, :, 0:3]

    Q = read_mat("./data/Q.mat")["Qnorm"].flatten()

    # convert data to ICs and normalise
    datasets = [x_test_0, x_test_1, x_test_2, x_test_3]

    processed_test_datasets = []

    for dataset in datasets:
        dataset = get_ICs(dataset, Q, minV, maxV)
        dataset = normalise_data(dataset, min_val, max_val)
        processed_test_datasets.append(dataset)
    processed_test_datasets = np.array(processed_test_datasets)

    # interpolate to get the needed cycles
    query_cycles = [200, 400, 600, 800, 1000]

    test_datasets = []
    test_labels = []

    for dataset in processed_test_datasets:
        interpolated_ICs, interpolated_degradation_modes = interpolate_data(
            dataset, y_test, cycles, query_cycles
        )
        test_datasets.append(interpolated_ICs)
        test_labels.append(interpolated_degradation_modes)

    test_datasets = np.array(test_datasets)
    test_labels = np.array(test_labels)

    get_pred(model, time_steps, test_datasets, test_labels)
