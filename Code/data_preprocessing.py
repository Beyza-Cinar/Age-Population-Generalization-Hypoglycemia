# imports
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
import datetime
import polars as pl



def gap_limited_interpolation(df_filtered, limit=6):
    """
    This function is used to impute missing values with linear interpolation based on a sepcified gap length.
    Parameters:
   - df_filtered is the original series of a defined column, 
   - limit is the number of consecutive values which should be imputed.
    Output: It returns a series of the specified column with linearly imputed values.
   """ 
    df_filtered = df_filtered.sort_index()
    # creates a group ID for consecutive NaNs
    is_nan = df_filtered.isna()
    not_nan = ~is_nan
    group_id = (not_nan != not_nan.shift()).cumsum()

    # gets all groups that are NaNs and their sizes
    nan_groups = is_nan.groupby(group_id, group_keys=False).transform("sum")

    # masks values to interpolate: NaNs and in small enough groups
    interpolate_mask = is_nan & (nan_groups < limit)

    # applies linear interpolation
    ts_interp = df_filtered.copy()
    interp_values = df_filtered.interpolate(method="linear")
    ts_interp[interpolate_mask] = interp_values[interpolate_mask]
    return ts_interp



# used for stineman interpolation
# source: https://github.com/jdh2358/py4science/blob/master/examples/extras/steinman_interp.py
def slopes(x, y):
    """
    Estimate the derivative y"(x) using a parabolic fit through three consecutive points.
    
    This method approximates slopes based on local parabolas and is numerically robust 
    for functions where abscissae (x) and ordinates (y) may differ in scale or units.

    Originally described by Norbert Nemec (2006), inspired by Halldor Bjornsson.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.full_like(y, np.nan)

    yp = np.full_like(y, np.nan)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        dydx = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)

    denom = dx[1:] + dx[:-1]
    valid = denom != 0

    numerator = dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.isfinite(denom) & (denom != 0), numerator / denom, 0.0)
    yp[1:-1][valid] = result[valid]

    yp[0] = 2 * dydx[0] - yp[1] if not np.isnan(yp[1]) else np.nan
    yp[-1] = 2 * dydx[-1] - yp[-2] if not np.isnan(yp[-2]) else np.nan

    return yp

# Stineman interpolation 
# original source: https://github.com/jdh2358/py4science/blob/master/examples/extras/steinman_interp.py
def stineman_interp(xi, x, y, yp=None):
    """
    Perform Stineman interpolation on known data points (x, y) at new locations xi.
    
    If derivative estimates yp are not supplied, they are estimated using `slopes(x, y)`.

    Returns interpolated values yi corresponding to xi.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)

    if yp is None:
        yp = slopes(x, y)
    else:
        yp = np.asarray(yp, dtype=float)

    if len(x) < 2:
        return np.full_like(xi, np.nan)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)

    idx = np.searchsorted(x[1:-1], xi)
    # protect out-of-bounds
    idx = np.clip(idx, 0, len(x) - 2)  

    sidx = s[idx]
    xidx = x[idx]
    yidx = y[idx]
    xidxp1 = x[idx + 1]

    yo = yidx + sidx * (xi - xidx)
    dy1 = (yp[idx] - sidx) * (xi - xidx)
    dy2 = (yp[idx + 1] - sidx) * (xi - xidxp1)
    dy1dy2 = dy1 * dy2

    dy1dy2_clean = np.nan_to_num(dy1dy2, nan=0.0, posinf=0.0, neginf=0.0)
    cond = np.sign(dy1dy2_clean).astype(int) + 1
    

    with np.errstate(divide="ignore", invalid="ignore"):
        denom_0 = (dy1 - dy2) * (xidxp1 - xidx)
        denom_2 = dy1 + dy2

        blend_0 = np.divide((2 * xi - xidx - xidxp1), denom_0,
                            out=np.zeros_like(dy1), where=denom_0 != 0)
        blend_2 = np.divide(1.0, denom_2,
                            out=np.zeros_like(dy1), where=denom_2 != 0)

        blend = np.choose(cond, [blend_0, np.zeros_like(dy1), blend_2])

    yi = yo + dy1dy2 * blend
    # converts any remaining inf to nan
    yi[np.isinf(yi)] = np.nan  

    return yi


# gap-limited Stineman interpolation
def interpolate_stineman_group(df_org, timestamp, value, llimit=6, ulimit=24, yp=None):
    """
    This function is used to impute missing values with Stineman interpolation based on a sepcified gap length.
    Parameters:
   - df_org is the original dataframe, 
   - timestamp is the name of the column with the timestamps, 
   - value is the name of the column which should be interpolated
   - llimit is the lower limit of the gap length
   - ulimit is the upper limit of the gap length
    Output: It returns the interpolated dataframe of the specified column with Stineman imputed values.
   """ 
    
    group = df_org.copy()
    group = group.sort_values(timestamp).copy()
    tf = (group[timestamp] - group[timestamp].min()).dt.total_seconds()
    val = group[value].copy()
    is_nan = val.isna().to_numpy()    

    # constraint
    if (~is_nan).sum() < 2:
        group[f"{value}_interp"] = val
        return group

    # labels NaN runs
    run_id = (is_nan != np.roll(is_nan, 1)).cumsum()
    run_id[0] = 1

    # computes run lengths
    run_lengths = pd.Series(is_nan).groupby(run_id).transform("sum").to_numpy()

    # gets run IDs that are fully eligible (between 4–12 NaNs only)
    run_id_series = pd.Series(run_id, index=val.index)
    valid_run_ids = run_id_series[(is_nan) & (run_lengths >= llimit) & (run_lengths < ulimit)].unique()

    # masks only those full runs
    mask = run_id_series.isin(valid_run_ids) & is_nan

    # interpolates with Stineman interpolation
    x_known = tf[~is_nan]
    y_known = val[~is_nan]
    x_interp = tf[mask]

    if x_interp.size > 0:
        y_interp = stineman_interp(x_interp, x_known, y_known)
        # we dont want unreasonable values
        #y_interp = np.clip(y_interp, 40, 500)
        y_interp = np.where((y_interp >= 40) & (y_interp <= 500), y_interp, np.nan)
        val.iloc[mask] = y_interp

    # rounds the output 
    group["value_interp"] = val.round(2)  

    # drops original columns
    group.drop(columns=[value], inplace=True)

    # renames the interpolated column
    group.rename(columns={"value_interp": value}, inplace=True)

    # returns the interpolated column
    return group



# generates classes for the hypoglycemia classification task
def class_generation(df_copy: pd.DataFrame, timestamp_col: str, start: int, end: int, class_number: int, col_name: str) -> pl.DataFrame: 
    """
    This function is used to assign labels.
    Parameters:
   - df_copy is the original pandas dataframe, 
   - timestamp_col is the name of the column with the timestamps, 
   - start is time in minutes of the minimum duration before hypogylcemia
   - end is time in minutes of the maximum duration before hypogylcemia
   - class_number is the class for the defined time range before hypoglycemia
    Output: It returns the dataframe with the assigned class for the spefied time range before hypoglycemia.
   """ 
    # first, the dataframe is converted to polars to increase efficacy
    df = pl.from_pandas(df_copy)
    # timestamps are sorted 
    df = df.sort(timestamp_col)
    
    # zero events are hypoglycemic data points
    # these are selected and the timestamps are stores as a series
    event_times = df.filter(pl.col(col_name) == 0).select(timestamp_col).to_series()

    # the start and end time of the time range before hypogylcemia are computed
    start_bounds = event_times - datetime.timedelta(minutes=start)
    end_bounds = event_times - datetime.timedelta(minutes=end)

    # the series is masked 
    mask = pl.Series([False] * df.height)

    # timestamps meeting the defined criteria of specified time bonds are identified 
    # moreover, the class should be -1 (not assinged as other classes before)
    for start_time, end_time in zip(start_bounds, end_bounds):
        window_mask = (
            (df[timestamp_col] > end_time) &
            (df[timestamp_col] <= start_time) &
            (df[col_name] == -1)
        )
        mask = mask | window_mask

    df = df.with_columns([
        pl.when(mask).then(class_number).otherwise(pl.col(col_name)).alias(col_name)
    ])

    # the dataframe is converted back to pandas 
    df_pd = df.to_pandas()
    # the dataframe with assigned classes is returned
    return df_pd

# data is normalized with minmax scaling
def normalize_data(df, value):
    """
    This function min-max scales the values of the specified colum.
    Parameters: 
        - df is the original dataframe
        - value is the name of the column which should be normalized
    """
    # copies the data 
    df_min_max_scaled = df.copy() 
    column = [value]
    # applies normalization techniques (min max scaling)
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())	
    # returns the data which was min-max scaled 
    return df_min_max_scaled


# generates time series with a sliding window appraoch of 2 hour lengths for the maindatabase 
# splits data into train, validation, and test
def extract_valid_windows_GLC(
    df_org: pd.DataFrame,
    timestamp_col: str = "ts",
    feature_col: str = "GlucoseCGM",
    class_col: str = "Class",
    expected_sample_count: int = 25,
    min_window_duration = np.timedelta64(2, "h")
):
    """
    This function is used genearte time series data and splits the data into train, validation, and test data.
    Parameters:
   - df_org is the original pandas dataframe, 
   - timestamp_col is the name of the column with the timestamps, 
   - feature_col is the name of the column which should be returned as a time series,
   - class_col is name of the column with the classes,
   - expected_sample_count is the number of allowed continuous values in a time series,
    - min_window_duration is the allowed continuous time of a time series
    Output: It returns a list including six separate list for X_train, X_val, X_test, Y_train, Y_val, and Y_test
   """ 
    # X_train, and Y_train are initialized as empty arrays
    X_train = []
    Y_train = []

    # converts pandas DataFrame to polars for faster execution
    df = pl.from_pandas(df_org)
    # sorts by timestamp to ensure chronological order
    df = df.sort(timestamp_col)

    # extracts timestamps and total row count
    timestamps = df[timestamp_col].to_numpy()
    n_rows = df.height

    # initializes empty lists for feature windows and labels
    windows = []
    labels = []

    # sets starting index for window generation
    start_idx = 0

    # iterates over all rows to construct valid windows
    for end_idx in range(n_rows):
        # moves start index if window exceeds allowed duration
        while timestamps[end_idx] - timestamps[start_idx] > min_window_duration:
            start_idx += 1

        # calculates number of samples in current window
        count = end_idx - start_idx + 1

        # checks if window meets required duration and sample count
        if (
            timestamps[end_idx] - timestamps[start_idx] >= min_window_duration and
            count == expected_sample_count
        ):
            # extracts current time window
            window_df = df.slice(start_idx, count)

            # skips window if missing values are present
            nulls = window_df.null_count().row(0)
            if all(count == 0 for count in nulls):
                # retrieves label from last row in window
                label = window_df[class_col].to_list()[-1]
                # allows only classes 0–4
                if label in {0, 1, 2, 3}:
                    # stores feature sequence and label
                    windows.append(window_df[feature_col].to_list())
                    labels.append(label)

    # checks if any valid windows were found
    if len(windows) > 0:
        # converts to numpy arrays
        windows = np.array(windows).reshape(-1, 1)
        labels = np.array(labels).reshape(-1, 1)

        # reshapes data for model input
        X_data = windows.reshape(-1, expected_sample_count, 1)
        Y_data = labels.reshape(-1, 1)

        # appends subject data to training sets
        X_train.append(X_data)
        Y_train.append(Y_data)

        # returns feature and label arrays
        return X_train, Y_train
    else:
        # returns nothing if no valid window exists
        return
    

# This function calls the function for generating time series sequences for each subject separately and returns them 
# as X and Y data in form of arrays
def create_X_Y(df, feature= "PtID", labels = "Class", sample_count= 25, hours = 2, modus = "h"):

    # generates time series with sliding window approach
    result = df.groupby(feature).apply(lambda g: extract_valid_windows_GLC(g, class_col = labels, expected_sample_count = sample_count,
    min_window_duration = np.timedelta64(hours, modus)))

    # filters out None values of subjects with insufficient data
    filtered_result = [item for item in result if item is not None]
    # unpacks the values
    X_data, Y_data = zip(*filtered_result)

    return X_data, Y_data



# flattens data 
def flatten_data(X, modus, axis_f = 1, shape_f = 25, dim = 1):
    """
    This function flattens the data returned from the time series generation function, since the window lists
    contain all windows of every subject.
    Parameters:
   - X is the original list, 
   - axis_f is the number of features, 
   - shape_f is the number of values in one window,
   - dim is the dimension,
    Output: It returns a the flattened array of the given list
   """ 
    array_data = [np.array(x) for x in X]  # convert all sublists to arrays
    flattened_data = np.concatenate(array_data, axis=axis_f)
    if modus == "input":
        flattened_data = flattened_data.reshape(-1,shape_f,dim)
    elif modus == "output":
        flattened_data = flattened_data.reshape(-1,dim)
    else:
        print("Modus must be either input or output.")
    print(flattened_data.shape) 
    return flattened_data



def remove_outliers_polars(df: pl.DataFrame, value: str, subject: str = "PtID") -> pl.DataFrame:
    """
    Removes outliers using IQR method per subject.
    Replaces values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] with nulls.
    Modus can be "glucose" or "vitals" to apply additional domain rules.
    """

    # Replace 0 with null first
    df = df.with_columns(
        pl.when(pl.col(value) == 0).then(None).otherwise(pl.col(value)).alias(value)
    )

    # Compute Q1, Q3, IQR per subject
    df = df.with_columns([
        pl.col(value).quantile(0.25).over(subject).alias("Q1"),
        pl.col(value).quantile(0.75).over(subject).alias("Q3")
    ])

    df = df.with_columns(
        (pl.col("Q3") - pl.col("Q1")).alias("IQR")
    )

    # Compute bounds
    df = df.with_columns([
        (pl.col("Q1") - 1.5 * pl.col("IQR")).alias("lower"),
        (pl.col("Q3") + 1.5 * pl.col("IQR")).alias("upper")
    ])

    # Define outlier condition based on modus
    outlier_condition = (
        (pl.col(value) < pl.col("lower")) |
        (pl.col(value) > pl.col("upper")) |
        (pl.col(value) < 40) |
        (pl.col(value) > 500)
    )
   
    # Replace outliers with nulls
    df = df.with_columns(
        pl.when(outlier_condition).then(None).otherwise(pl.col(value)).alias(value)
    )

    # Drop temporary helper columns
    return df.drop(["Q1", "Q3", "IQR", "lower", "upper"])


 #### DATA PREPROCESSING ####

def bfill_hba1c_all(static_data: pl.DataFrame) -> pl.DataFrame:
    return static_data.with_columns([
        pl.col("Hba1c").backward_fill(limit=25920).over("PtID").alias("Hba1c")
    ])

def ffill_height_all(static_data: pl.DataFrame) -> pl.DataFrame:
    return static_data.with_columns([
        pl.col("HeightCm").forward_fill(limit=105120).over("PtID").alias("HeightCm")
    ])

def ffill_weight_all(static_data: pl.DataFrame) -> pl.DataFrame:
    return static_data.with_columns([
        pl.col("WeightKg").forward_fill(limit=105120).over("PtID").alias("WeightKg")
    ])

def impute_height_adults(static_data: pl.DataFrame) -> pl.DataFrame:
    return (
        static_data
        .with_columns([
            # Keeps Hba1c values only for SHD rows; else set to null
            pl.when((pl.col("AgeGroup") == 7) & (pl.col("AgeGroup") == 8))
              .then(pl.col("HeightCm"))
              .otherwise(None)
              .alias("HeightCm_adult")
        ])
        .with_columns([
            # Applies backward fill within each PtID group
            pl.col("HeightCm_adult").forward_fill().over("PtID").alias("HeightCm_adult_filled")
        ])
        .with_columns([
            # Uses filled values where Database == SHD, else keep original
            pl.when((pl.col("AgeGroup") == 7) & (pl.col("AgeGroup") == 8))
              .then(pl.col("HeightCm_adult_filled"))
              .otherwise(pl.col("HeightCm"))
              .alias("HeightCm")
        ])
        .drop(["HeightCm_adult", "HeightCm_adult_filled"])  
    )


def impute_weight_adults(static_data: pl.DataFrame) -> pl.DataFrame:
    return (
        static_data
        .with_columns([
            pl.when((pl.col("AgeGroup") == 7) & (pl.col("AgeGroup") == 8))
              .then(pl.col("WeightKg"))
              .otherwise(None)
              .alias("WeightKg_adult")
        ])
        .with_columns([
            # Applies backward fill within each PtID group
            pl.col("WeightKg_adult").forward_fill().over("PtID").alias("WeightKg_adult_filled")
        ])
        .with_columns([
            pl.when((pl.col("AgeGroup") == 7) & (pl.col("AgeGroup") == 8))
              .then(pl.col("WeightKg_adult_filled"))
              .otherwise(pl.col("WeightKg"))
              .alias("WeightKg")
        ])
        .drop(["WeightKg_adult", "WeightKg_adult_filled"])  
    )





# This functions adds previous datapoints as features
def add_time_features(df: pl.DataFrame, feature: str = "GlucoseCGM") -> pl.DataFrame:
    df =  df.sort(["PtID", "ts"])
    
    df = df.with_columns([
        # Differences
        (pl.col(feature) - pl.col(feature).shift(1)).alias("diff_5"),
        (pl.col(feature) - pl.col(feature).shift(2)).alias("diff_10"),
        (pl.col(feature) - pl.col(feature).shift(4)).alias("diff_20"),
        # First-order difference
        pl.col(feature).diff().alias("diffs"),
    ])
    return df.drop("diffs")


# This function computes statistical features from rolling windows
def add_statistical_features(df: pl.DataFrame, column: str, window: int, suffix: str) -> pl.DataFrame:
    df =  df.sort(["PtID", "ts"])
    return df.with_columns([
        pl.col(column).rolling_mean(window).alias(f"mean_{suffix}"),
        pl.col(column).rolling_std(window).alias(f"std_{suffix}"),
        pl.col(column).rolling_min(window).alias(f"min_{suffix}"),
        pl.col(column).rolling_max(window).alias(f"max_{suffix}"),
        ((pl.col(column) < 70).cast(pl.Int8).rolling_mean(window) * 100).alias(f"time_below70_{suffix}"),
        ((pl.col(column) > 200).cast(pl.Int8).rolling_mean(window) * 100).alias(f"time_above200_{suffix}"),

        # neu
        (pl.col(column) - pl.col(column).shift(window)).alias(f"diff_{suffix}"),
        # Slopes
        (pl.col(column) - (pl.col(column).shift(window)) / window).alias(f"slope_{suffix}"),
    ])


# This function adds the statistical metrics for defined windows
def apply_all_metrics(group: pl.DataFrame) -> pl.DataFrame:
    # Starts with raw group
    df = group

    # Adds multiple rolling windows
    for window, suffix in [(6, "30min"), (12, "1hr"), (24, "2hr")]:
        df = add_statistical_features(df, column="GlucoseCGM", window=window, suffix=suffix)

    return df


def convert_age_strings(df: pl.DataFrame, column: str = "Age") -> pl.DataFrame:
    return df.with_columns([
        # Cleans strings: remove "yrs", spaces
        pl.col(column)
        .cast(str)
        .str.replace_all("yrs", "")
        .str.replace_all(r"\s+", "")
        .alias("Age_clean")
    ]).with_columns([
        # Extract start and end as floats
        pl.col("Age_clean").str.extract(r"^(\d+)", 1).cast(pl.Float64, strict=False).alias("age_start"),
        pl.col("Age_clean").str.extract(r"-(\d+)", 1).cast(pl.Float64, strict=False).alias("age_end")
    ]).with_columns([
        # Computes midpoint if range exists; else keep age_start
        pl.when(pl.col("age_end").is_not_null())
          .then((pl.col("age_start") + pl.col("age_end")) / 2)
          .otherwise(pl.col("age_start"))
          .alias("Age_numeric")
    ]).drop(["Age_clean", "age_start", "age_end"])


# Assigns the definied age groups
def assign_age_group_from_string_ranges(df: pl.DataFrame) -> pl.DataFrame:
    
    column = pl.col("Age_numeric")  

    return df.with_columns([
        pl.when(column <= 13).then(0)
        .when(column <= 20).then(1)
        .when(column <= 44).then(2)
        .when(column <= 100).then(3)
        .otherwise(4)
        .alias("AgeGroup")
    ])


# Converts all sublists to arrays
def flatten(subject):
    array_data = [np.array(x) for x in subject]  
    flattened_data = np.concatenate(array_data, axis=0)
    return flattened_data