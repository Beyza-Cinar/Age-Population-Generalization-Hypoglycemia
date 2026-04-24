"""
This file provides functions to read the seperate datasets, store them in a list and integrate them into three separate datasets.
"""


# necessary imports
import pandas as pd
import glob
import os
import csv
import re
import numpy as np
import polars as pl

def df_resample(df_org, timestamp, frequency, mode, fillid, value):
    """
    This function is used to resample the dataset to the same frequency.
    Parameters:
   - df_org is the original dataset, 
   - timestamp is the name of the column with the timestamps
   - frequency is the target frequency
   - the mode—either "glucose" or "vitals"—specifies whether PtID fillforward is necessary; heart rate data does not require this
   - fillid is the name of the column which needs fillforward to impute produced missing values
   - value-either "glucose" or "heartrate"-is the column name which should be converted into integer values 
    Output: It returns the resampled dataframe with no missing values in the PtID column and glucose/heartrate values as float values
   """ 
    df = df_org.copy()
    # if the mode is set to "glucose", the timestamps are first sorted, then rounded, and finally resampled to match the target frequency
    if mode == "glucose":
        df = df.sort_values(by=timestamp)
        df[timestamp] = df[timestamp].dt.round("5min")
        # rounding can induce duplicates which need to be removed to enable resampling
        df = df.drop_duplicates(subset=[timestamp])
        # remove nan values in the timestamp column 
        df = df.dropna(subset=[timestamp])
        # the timestamps column is set to be the index 
        df = df.set_index(timestamp)
        # based on the index, the dataframe is resampled to the target frequency
        df = df.resample(frequency, origin="start").asfreq()
        # the index is resetted 
        df = df.reset_index()
        # missing values which occured due to resampling but are essential are feedforward filled 
        df[fillid] = df[fillid].fillna(method="ffill")
    # if the mode is set to "vitals", the timestamps are only rounded to the target frequency
    elif mode == "vitals":
        df[timestamp] = df[timestamp].dt.round(frequency)
        df = df.drop_duplicates(subset=[timestamp])
    # if a wrong input is given, a warning is outputted
    else:
        print("Mode can be either glucose or vitals")
    # finally, the column with the specified values is converted to numericals (floats)
    df[value] = pd.to_numeric(df[value], errors="coerce")
    # the resampled dataframe is returned
    return df


def fill_gaps_sampling(df, timestamp, subject_id, glucose, frequency, fillmin=15,  fillvalues = True):
    """
    This function applies feedforward filling to glucose values missing as a result of undersampling.
    Parameters:
    - df is the original dataframe
    - timestamp is the name of the column with the timestamps
    - subjects id is the name of the column with the subject ids
    - glucose is the name of the column with the glucose measurements
    - fillmin is the original frequency of the dataset; by default it is set to 15 minutes
    """ 
    if fillvalues == True:
        # first, the timestamp column of the original dataframe is sorted and then copied
        sorted_df = df.sort_values(timestamp).copy()
        # continuous true glucose measurements are identified having a time difference of the dataset"s original frequency
        sorted_df["time_diff"] = sorted_df[timestamp].diff()
        sorted_df["gap"] = sorted_df["time_diff"] > pd.Timedelta(minutes=fillmin)
        # consecutive glucose measurements are grouped and assigned a group id 
        sorted_df["group_id"] = sorted_df["gap"].cumsum()

        # an empty list is intialized which will store individually resampled groups 
        resampled_groups = []

        # for each group of consecutive measurements, the timestamp is undersampled to 5 minutes and occuring gaps are filled with the fillforward method
        for _, group in sorted_df.groupby("group_id"):
            # to resample the group, the function "df_resample()" is called
            group_resampled = df_resample(group, timestamp = timestamp, frequency= frequency, mode="glucose", fillid = subject_id, value = glucose)
            group_resampled[glucose] = group_resampled[glucose].ffill()
            # the resampled group without gaps is appended to the initialized list
            resampled_groups.append(group_resampled)

        # all groups are concatenated to one dataframe
        resampled_df = pd.concat(resampled_groups)
        # finally, the whole dataset is resampled to 5 minute intervals
        resampled_df = df_resample(resampled_df, timestamp = timestamp, frequency= frequency, mode="glucose", fillid = subject_id, value = glucose)
        # the resamoked dataset is returned
        return resampled_df.reset_index()
    else: 
        sorted_df = df.sort_values(timestamp).copy()
        resampled_df = df_resample(sorted_df, timestamp = timestamp, frequency= frequency, mode="glucose", fillid = subject_id, value = glucose)
        # the resamoked dataset is returned
        return resampled_df.reset_index()

def detect_best_separator(file_path, sample_size=10000):
    """ 
    This function finds the best seperator enable the automatic read of the dataset.
    Parameter: 
    - file_path is the path of the file 
    - sample size is the amount of characters which should be read to find the sepatator
    Output: best found seperator is outputted
    """
    # possible encodings are defined
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin1"]
    # possible delimeters are defined
    delimiters = [",", ";", "\t", "|"]

    # the file is opened with the correct encoding by trying each encoding
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                # the file is opened to read
                sample = f.read(sample_size)
                # if UTF-16, the delimeters are count to find the best option instead of using the Sniffer function
                if enc == "utf-16":
                    counts = {d: sample.count(d) for d in delimiters}
                    # the delimeter with the majority of counts is chosen 
                    best_guess = max(counts, key=counts.get)
                    # if the majority is larger than 0 it is returned as the best guess
                    if counts[best_guess] > 0:
                        return best_guess
                    continue
                # else, try the Sniffer function which automatically returns the delimeter
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
                    return dialect.delimiter
                # if an error occurs, fallback: use count-based detection
                except csv.Error:
                    counts = {d: sample.count(d) for d in delimiters}
                    best_guess = max(counts, key=counts.get)
                    if counts[best_guess] > 0:
                        return best_guess
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not detect delimiter or unsupported encoding.")




def smart_read(file_path, skip = 0):
    """
    This function automatically reads a file into a pandas DataFrame based on its extension.
    It automatically detects the delimiter for .csv and .txt files from function "detect_best_separator()".
    Parameter: 
    - file_path is the path of the file
    - skip is the number of rows which should be skipped 
    Output: a pandas dataframe is outputted
    """

    # extracts the extension of the file 
    ext = os.path.splitext(file_path)[1].lower()

    # checks if the file is an excel file
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    # checks if the file is a csv or txt file
    elif ext in [".csv", ".txt"]:
        # based on the returned best separator, the file is read
        sep = detect_best_separator(file_path)
        # try with normal encoding
        try:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", skiprows=skip)
        # if an error occurs, try with the utf-16 encoding 
        except:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", encoding="utf-16", skiprows=skip)
    # if still an error occurs, output warning 
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # returns the pandas dataframe
    return df




def detect_sample_rate(df, time_col=None, expected_rates=(5, 10, 15)):
    """
    This functions detects the sample rate of the glucose measurements.
    A frequency of 5, 10, and 15 minutes are allowed which is typical for CGM devices.
    Parameter: df is the original dataframe 
    Output: most common frequency is outputted
    """
    # extracts time column or index
    times = pd.to_datetime(df[time_col]) if time_col else pd.to_datetime(df.index)
    times = times.sort_values()

    # computes time differences in minutes
    deltas = times.diff().dropna().dt.total_seconds() / 60  # in minutes

    # rounds to nearest whole minute
    deltas_rounded = deltas.round().astype(int)

    # finds the most common interval
    most_common = deltas_rounded.value_counts().idxmax()

    # matches to known expected rates
    if most_common in expected_rates:
        # prints the most common frequency in monutes 
        return f"{most_common} min"
    else:
        return "Unknown"
    


def read_data(read_all = True, frequency = "5min"):
    """
    This function reads the datasets individually and returns them as a list of dataframes.
    Parameter: read_all can be True or False. If true, all datasets are read and returned. If false, only the set of restricted datasets are read.
    Output: returns a list of pandas dataframes
    """
    def df_granada():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_granada = smart_read("Data/datasets_for_T1D/granada/T1DiabetesGranada/glucose_measurements.csv")
        # column names are renamed for semantic equality
        df_granada = df_granada.rename(columns={"Measurement": "GlucoseCGM", "Patient_ID": "PtID"})

        # all datasets should keep the same format of date and time -> combine both and convert to datetime
        df_granada["ts"] = pd.to_datetime(df_granada["Measurement_date"] + " " + df_granada["Measurement_time"])
        # undersamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_granada = df_granada.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM"))                                                               

        # reads the dataframe with demographics
        df_granada_info = smart_read("Data/datasets_for_T1D/granada/T1DiabetesGranada/Patient_info.csv")
        # converts timstamps to datetime
        df_granada_info["Initial_measurement_date_date"] = pd.to_datetime(df_granada_info["Initial_measurement_date"])
        # reduces the columns to only important columns
        df_granada_info = df_granada_info[["Sex", "Birth_year", "Patient_ID"]]
        # column names are renamed for semantic equality
        df_granada_info = df_granada_info.rename(columns={"Patient_ID": "PtID"})
        # merges both dataframes
        df_granada = pd.merge(df_granada, df_granada_info, on="PtID", how="left")
        # computes the age based on the birth year and the datetime of measurement
        df_granada["Age"] = df_granada["ts"].dt.year - df_granada["Birth_year"]
        # removes the "Birth_year" column
        df_granada = df_granada.drop(["Birth_year"], axis=1)

        # reads the dataframe with biochemical parameters to extract Hbac1
        df_granada_BioP = pd.read_csv("Data/datasets_for_T1D/granada/T1DiabetesGranada/Biochemical_parameters.csv")
        # extracts only Hbac1 data
        df_granada_BioP = df_granada_BioP[(df_granada_BioP["Name"] == "Glycated hemoglobin (A1c)")]
        # renames values
        df_granada_BioP = df_granada_BioP.rename(columns={"Value": "Hba1c", "Patient_ID" : "PtID", "Reception_date" : "Measurement_date"})
        df_granada_BioP = df_granada_BioP[["PtID", "Measurement_date", "Hba1c"]]
        # extracts only the date of the maindatabase from the timstamp column
        df_granada["Measurement_date"] = pd.to_datetime(df_granada["ts"].dt.date)
        df_granada_BioP["Measurement_date"] = pd.to_datetime(df_granada_BioP["Measurement_date"])

        # merges Hbac1 with the main database with a left outer join
        df_granada = pd.merge(df_granada, df_granada_BioP, on=["PtID", "Measurement_date"], how="left")

        # adds the database name to the patient ID to enable reidentification 
        df_granada["PtID"] = df_granada["PtID"].astype(str) + "_T1DGranada"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_granada["Database"] = "T1DGranada"
        return df_granada


    def df_diatrend():
        # path to the single CGM datasets of each subject
        file_paths_diatrend = glob.glob("Data/datasets_for_T1D/Diatrend/DiaTrendAll/*.xlsx") 

        # initializes an empty list to store the dataframes
        df_list_diatrend = []

        # loops through each file
        for idx, file in enumerate(file_paths_diatrend, start=1):
            # reads the file with the "smart_read()" function
            df = smart_read(file)  
            # extracts the PtID from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # adds the extracted ID to the "PtID" column
            df["PtID"] = str(name)
            # adds the dataframe to the list
            df_list_diatrend.append(df)

        # concatenates all dataframes into one dataframe
        df_diatrend = pd.concat(df_list_diatrend, ignore_index=True)

        # column names are renamed for semantic equality
        df_diatrend = df_diatrend.rename(columns={"mg/dl": "GlucoseCGM"})
        # converts timstamps to datetime
        df_diatrend["ts"] = pd.to_datetime(df_diatrend["date"], format="%Y-%m-%d %H:%M:%S")
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_diatrend = df_diatrend.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe including demographics
        df_diatrend_info = smart_read("Data/datasets_for_T1D/Diatrend/SubjectDemographics_3-15-23.xlsx") 
        # column names are renamed for semantic equality
        df_diatrend_info["Sex"] = df_diatrend_info["Gender"].replace({"Male": "M", "Female": "F"})
        df_diatrend_info = df_diatrend_info.rename(columns={"Hemoglobin A1C":"Hba1c"})

        # adds a "PtID" column storing the SubjectIDs
        df_diatrend_info["PtID"] = "Subject" + df_diatrend_info["Subject"].astype(str)
        # copies dataset for later use of Hba1c value
        df_diatrend_temp_hba1c = df_diatrend_info.copy()
        # drops hba1c value from dataset
        df_diatrend_info = df_diatrend_info.drop("Hba1c", axis=1)

        # merges dataframes with demographics and CGM data
        df_diatrend = pd.merge(df_diatrend, df_diatrend_info, on=["PtID"], how="left")


        # copies original age to preserve string version
        df_diatrend["age_str"] = df_diatrend["Age"]
        # tries to convert to numeric where possible 
        df_diatrend["age_int"] = pd.to_numeric(df_diatrend["Age"], errors="coerce")
        df_diatrend["initial_date"] = df_diatrend.groupby("PtID")["ts"].transform("min")
        # calculates birth year where age_int is available
        df_diatrend["birthyear"] = df_diatrend["initial_date"].dt.year - df_diatrend["age_int"] 
        # calculates age from birth year (re-checking with timestamp)
        df_diatrend["Age_from_byear"] = df_diatrend["ts"].dt.year - df_diatrend["birthyear"]
        # creates final age column
        df_diatrend["final_age"] = df_diatrend.groupby("PtID").apply(lambda group: group["Age_from_byear"].fillna(group["age_str"])).reset_index(level=0, drop=True)
        # drops the old age column
        df_diatrend = df_diatrend.drop("Age", axis=1)
        # creates a new age colum
        df_diatrend["Age"] = df_diatrend["final_age"]
        # converts float numbers into integers
        df_diatrend["Age"] = df_diatrend["Age"].apply(lambda x: int(x) if isinstance(x, float) else x)
        df_diatrend["initial_date_only"] = df_diatrend["ts"].dt.date.astype(str) 

        df_diatrend_temp_hba1c = df_diatrend_temp_hba1c[["PtID","Hba1c"]]
        first_dates = df_diatrend.groupby("PtID")["ts"].min().reset_index()
        first_dates.rename(columns={"ts": "initial_date_only"}, inplace=True)

        # merges first_dates with PtIds
        df_diatrend_temp_hba1c = df_diatrend_temp_hba1c.merge(first_dates, on="PtID", how="left")
        df_diatrend_temp_hba1c["initial_date_only"] = df_diatrend_temp_hba1c["initial_date_only"].dt.date.astype(str) 
        df_diatrend = pd.merge(df_diatrend, df_diatrend_temp_hba1c, on=["PtID", "initial_date_only"], how="left")


        # adds the database name to the patient ID to enable reidentification 
        df_diatrend["PtID"] = df_diatrend["PtID"].astype(str) + "_DiaTrend"

        df_diatrend = df_diatrend[["ts", "GlucoseCGM", "PtID", "Gender", "Race",
            "Hba1c", "Sex","Age"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_diatrend["Database"] = "DiaTrend"
        return df_diatrend

    def df_city():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_city = smart_read("Data/datasets_for_T1D/CITYPublicDataset/Data Tables/DeviceCGM.txt")
        # column names are renamed for semantic equality
        df_city = df_city.rename(columns={"Value": "GlucoseCGM"})

        # differentiates between CGM glucose and finger prick glucose
        df_city["mGLC"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "Calibration")
        df_city["GlucoseCGM"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "CGM")

        # converts timstamps to datetime
        df_city["ts"] = pd.to_datetime(df_city["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_city = df_city.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe including demographics data
        df_city_screen = smart_read("Data/datasets_for_T1D/CITYPublicDataset/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_city_screen = df_city_screen[["PtID", "Sex", "Ethnicity", "Race", "DiagDt", "DiagAge"]]
        # fill nan values in diagage column with the approximate age 


        # reads dataframe including age data
        df_city_age = smart_read("Data/datasets_for_T1D/CITYPublicDataset/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_city_age = df_city_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merges dataframes of sex and age 
        df_city_info = pd.merge(df_city_screen, df_city_age, on=["PtID"], how="left")

        # merges dataframes of demorgaphics and CGM data
        df_city = pd.merge(df_city, df_city_info, on=["PtID"], how="left")

        # computes the birthyear from the age at measurement day
        df_city["EnrollDt"] = pd.to_datetime(df_city["EnrollDt"])
        df_city["Birthyear"] = df_city["EnrollDt"].dt.year - df_city["AgeAsOfEnrollDt"]
        # computes the age of each year based on the birthyear
        df_city["Age"] = df_city["ts"].dt.year - df_city["Birthyear"]

        # adds the database name to the patient ID to enable reidentification 
        df_city["PtID"] = df_city["PtID"].astype(float).astype(str) + "_CITY"

        # extracts the Hbac1 value from the dataframe 
        df_city_hbac1 = smart_read("Data/datasets_for_T1D/CITYPublicDataset/Data Tables/DiabLocalHbA1c.txt")
        df_city_hbac1 = df_city_hbac1[["PtID", "HbA1cTestDt", "HbA1cTestRes"]]

        # adds the database name to the patient ID to enable reidentification and enabling the merging based on the patient ID
        df_city_hbac1["PtID"] = df_city_hbac1["PtID"].astype(float).astype(str) + "_CITY"
        # extracts the dates of all timestamps and stores them as a new column 
        df_city["HbA1cTestDt"] = pd.to_datetime(df_city["ts"].dt.date)
        # converts the date of the HbA1c measurement to datetime
        df_city_hbac1["HbA1cTestDt"] = pd.to_datetime(df_city_hbac1["HbA1cTestDt"])

        # merges Hbac1 with the main database with a left outer join
        df_city = pd.merge(df_city, df_city_hbac1, on=["PtID", "HbA1cTestDt"], how="left")

        # renames column names for semantic equality
        df_city  = df_city.rename(columns={"HbA1cTestRes": "Hba1c"})


        # finds the first timestamp per subject (initial date)
        df_city["initial_date"] = df_city.groupby("PtID")["ts"].transform("min")
        # calculates difference in days and convert to weeks (as integer)
        df_city["weeks_passed"] = ((df_city["ts"] - df_city["initial_date"]).dt.days // 7)
        # creates a label as X week visit to match the column type of the DiabPhysExam dataframe
        df_city["Visit"] = df_city["weeks_passed"].astype(str) + " week visit"

        df_city_physic = smart_read("Data/datasets_for_T1D/CITYPublicDataset/Data Tables/DiabPhysExam.txt")

        # convert weight to kg for semantic equality
        df_city_physic["WeightKg"] = df_city_physic["Weight"]
        df_city_physic.loc[df_city_physic["WeightUnits"] == "lbs", "WeightKg"] = df_city_physic.loc[df_city_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237

        # convert height to cm for semantic equality
        df_city_physic["HeightCm"] = df_city_physic["Height"]
        df_city_physic.loc[df_city_physic["HeightUnits"] == "in", "HeightCm"] = df_city_physic.loc[df_city_physic["HeightUnits"] == "in", "Height"] * 2.54

        df_city_physic = df_city_physic.drop(["RecID", "ParentLoginVisitID", "PhysExamNotDone", "WeightUnk", "HeightUnk", "Weight", "WeightUnits", "Height", "HeightUnits", "BldPrUnk", "PEHeartRtUnk", "TempUnk","FingStkBGUnk", "PEAbnormal" ], axis=1)


        df_city_physic["Visit"] = df_city_physic["Visit"].replace("Screening", "0 week visit")
        # adds the database name to the patient ID to enable reidentification and enabling the merging based on the patient ID
        df_city_physic["PtID"] = df_city_physic["PtID"].astype(float).astype(str) + "_CITY"
        # merges with the main database
        df_city = pd.merge(df_city, df_city_physic, on=["PtID", "Visit"], how="left")

        df_city["LOfDiabDiag"] = df_city["Age"] - df_city["DiagAge"]

        # removes unncessary columns
        df_city = df_city[["ts", "PtID", "DeviceDtTm","GlucoseCGM", "Sex",
            "Ethnicity", "Race", "DiagAge", "LOfDiabDiag", "Age", "HbA1cTestDt", "Hba1c", "WeightKg", "HeightCm"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_city["Database"] = "CITY"
        return df_city

    def df_dclp():
        # this dataset has two different CGM records which are read seperately with the "smart_read()" function
        df_dclp = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DexcomClarityCGM_a.txt")
        # reduces the columns to only important columns
        df_dclp = df_dclp[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adj"]]

        df_dclp_other = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/OtherCGM_a.txt")
        # reduces the columns to only important columns
        df_dclp_other = df_dclp_other[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adjusted"]]

        # column names are renamed for semantic equality
        df_dclp_other.rename(columns= {"DataDtTm_adjusted": "DataDtTm_adj"})


        df_dclp_diasend = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DiasendCGM_a.txt")
        # reduces the columns to only important columns
        df_dclp_diasend = df_dclp_diasend[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adjusted"]]

        # column names are renamed for semantic equality
        df_dclp_diasend.rename(columns= {"DataDtTm_adjusted": "DataDtTm_adj"})

        # merges both dataframes
        df_DCLP = pd.concat([df_dclp, df_dclp_other, df_dclp_diasend])

        # converts timstamps to datetime
        df_DCLP["ts"] = pd.to_datetime(df_DCLP["DataDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DCLP = df_DCLP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "CGM"))

        # reads dataframes of demographics 
        df_dclp_screen = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DiabScreening_a.txt")
        df_dclp_screen = df_dclp_screen[["PtID", "AgeAtEnrollment", "Gender", "Ethnicity", "Race", "RaceDs", "DiagDt", "DiagAge"]]
        # merges both dataframes
        df_DCLP = pd.merge(df_DCLP, df_dclp_screen, on=["PtID"], how="left")


        # reads dataframes of the physical examination 
        df_dclp_physic = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DiabPhysExam_a.txt")
        # convert weight to kg for semantic equality
        df_dclp_physic["WeightKg"] = df_dclp_physic["Weight"]
        df_dclp_physic.loc[df_dclp_physic["WeightUnits"] == "lbs", "WeightKg"] = df_dclp_physic.loc[df_dclp_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237
        # convert height to cm for semantic equality
        df_dclp_physic["HeightCm"] = df_dclp_physic["Height"]
        df_dclp_physic.loc[df_dclp_physic["HeightUnits"] == "in", "HeightCm"] = df_dclp_physic.loc[df_dclp_physic["HeightUnits"] == "in", "Height"] * 2.54
        # keeps only important columns
        df_dclp_physic = df_dclp_physic[["PtID", "BldPrSys", "BldPrDia", "PEHeartRt", "Temp", "TempUnits", "FingStkBG", "FingStkBGUnits", "WeightKg", "HeightCm"]]
        df_DCLP = pd.merge(df_DCLP, df_dclp_physic, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_DCLP["initial_date"] = df_DCLP.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_DCLP["Birthyear"] = df_DCLP["initial_date"].dt.year - df_DCLP["AgeAtEnrollment"] 
        # computes the age of each year based on the birthyear
        df_DCLP["Age"] = df_DCLP["ts"].dt.year - df_DCLP["Birthyear"]

        df_DCLP["LOfDiabDiag"] = df_DCLP["Age"] - df_DCLP["DiagAge"]

        # reads the dataframe with the hbac1 results
        df_dclp_hbac1 = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DiabLocalHbA1c_a.txt")
        # kepps only important data
        df_dclp_hbac1 = df_dclp_hbac1[["PtID", "HbA1cTestDt", "HbA1cTestRes"]]
        # extracts the date of the measurement timestamp
        df_dclp_hbac1["date"] = pd.to_datetime(pd.to_datetime(df_dclp_hbac1["HbA1cTestDt"]).dt.date)
        # extracts only the date of the timestamp in the original database as a new column
        df_DCLP["date"] = pd.to_datetime((df_DCLP["ts"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_DCLP = pd.merge(df_DCLP, df_dclp_hbac1, on=["PtID", "date"], how="left")


        # reads lab based Hba1c results
        df_dclp_hbac1_sample = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/SampleResults_a.txt")
        # keeps only important data
        df_dclp_hbac1_sample = df_dclp_hbac1_sample[df_dclp_hbac1_sample["ResultName"] == "GLYHB"]
        df_dclp_hbac1_sample = df_dclp_hbac1_sample[["PtID", "Value", "CollectionDt"]]
        # renames values 
        df_dclp_hbac1_sample = df_dclp_hbac1_sample.rename(columns={"Value": "LabHba1c"})
        df_dclp_hbac1_sample["date"] = pd.to_datetime(pd.to_datetime(df_dclp_hbac1_sample["CollectionDt"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_DCLP = pd.merge(df_DCLP, df_dclp_hbac1_sample, on=["PtID", "date"], how="left")
        df_DCLP["HbA1cTestRes"] = df_DCLP["HbA1cTestRes"].fillna(df_DCLP["LabHba1c"])

       
        # column names are renamed for semantic equality
        df_DCLP = df_DCLP.rename(columns={"CGM": "GlucoseCGM", "Gender" : "Sex", "HbA1cTestRes" : "Hba1c"}) 
        # adds the database name to the patient ID to enable reidentification 
        df_DCLP["PtID"] = df_DCLP["PtID"].astype(str) + "_DLCP3"
        # reduces database to important values
        df_DCLP = df_DCLP[["ts", "PtID", "GlucoseCGM", "DataDtTm_adj", "AgeAtEnrollment", "Sex", "Ethnicity", "Race",
            "RaceDs","DiagAge", "LOfDiabDiag", "WeightKg", "HeightCm", "Age", "HbA1cTestDt", "Hba1c", "LabHba1c",
            "CollectionDt"]]
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_DCLP["Database"] = "DLCP3"
        return df_DCLP
    
    def df_hupa():
        # path to the base directory
        base_path_hupa = "Data/datasets_for_T1D/HUPA-UCM/Raw_Data"

        # initilaizes lists to store all dataframes of heartrate and glucose
        all_data_HR_h = []
        all_data_GLC_h = []


        # loops through each subject directory 
        for subject_id_h in os.listdir(base_path_hupa):

            # reads all files including CGM readings
            person_path_h = os.path.join(base_path_hupa, subject_id_h)
            # skips if not a directory
            if not os.path.isdir(person_path_h):
                continue
            # combines the path name of the person and the folder
            for folder_h in os.listdir(person_path_h):
                folder_path_h = os.path.join(person_path_h, folder_h)

                # skips if not a directory
                if not os.path.isdir(folder_path_h):
                    continue
                # loops through each file 
                for file in os.listdir(folder_path_h):
                    
                    # skips if not a directory
                    if not os.path.isdir(folder_path_h):
                        continue
                    
                    # if file contains heart and is a csv file, the file is read
                    if file.endswith(".csv") and "heart" in file:
                        # combines the paths 
                        file_path_hr_h = os.path.join(folder_path_h, file)

                        try:
                            # reads the dataframe with the heartrate measurements with the "smart_read()" function 
                            hupa_hr = smart_read(file_path_hr_h)
                            # adds a PtID
                            hupa_hr["PtID"] = subject_id_h
                            # extracts the date from the filename
                            match = re.search(r"([\d]{4}-[\d]{2}-[\d]{2})", file_path_hr_h)
                            if match:
                                date_str = match.group(1)
                                # adds the date 
                                hupa_hr["date"] = date_str
                                # converts the time to a string
                                hupa_hr["Time"] = hupa_hr["Time"].astype(str)
                                # combines the date with the time and converts to datetime
                                hupa_hr["ts"] = pd.to_datetime(hupa_hr["date"] + " " + hupa_hr["Time"])
                                # column names are renamed for semantic equality
                                hupa_hr = hupa_hr.rename(columns={"Heart Rate" : "HR"})
                                # reduces the columns to only important columns
                                hupa_hr = hupa_hr[["ts", "PtID", "HR"]]
                            else:
                                print("No date found in the file path.")
                            # adds all files containing HR to the same list
                            all_data_HR_h.append(hupa_hr)

                        except Exception as e:
                            print(f"Failed to read {file_path_hr_h}: {e}")

                    # if file contains free style sensor and is a csv file, read the file; these files contain CGM measurements in 15 minute intervals
                    elif file.endswith(".csv") and "free_style_sensor" in file:
                        # joins the paths
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        # subjects 25-28 have a different schema , thus these are loaded differently
                        if re.search(r"2[5-8]P", subject_id_h):
                            try:
                                # dataframes with the CGM measurements are read with the "smart_read()" function
                                hupa_glc = smart_read(file_path_glc_h, skip=2) 
                                # adds Subjects ID
                                hupa_glc["PtID"] = subject_id_h
                                # column names are renamed for semantic equality
                                hupa_glc = hupa_glc.rename(columns={"Sello de tiempo del dispositivo": "ts", "Historial de glucosa mg/dL" : "Historic Glucose", 
                                        "Escaneo de glucosa mg/dL": "Scan Glucose", "Tira reactiva para glucosa mg/dL": "MGlucose"})
                                # converts timstamps to datetime
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # historic glucose is replaced with aligning scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                # reduces the columns to only important columns
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # resamples to 5 minute intervals to have unifrom sample rate
                                hupa_glc = df_resample(hupa_glc, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM")                                                               
                                
                                # adds all glucose data into one list 
                                all_data_GLC_h.append(hupa_glc)

                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}")
                        else: 
                            # reads remaining subjects
                            try: 
                                # dataframes with the CGM measurements are read with the "smart_read()" function
                                hupa_glc = smart_read(file_path_glc_h, skip=1) 
                                # adds Subject ID 
                                hupa_glc["PtID"] = subject_id_h
                                # column names are renamed for semantic equality
                                hupa_glc = hupa_glc.rename(columns={"Hora": "ts", "Histórico glucosa (mg/dL)" : "Historic Glucose", 
                                        "Glucosa leída (mg/dL)": "Scan Glucose", "Glucosa de la tira (mg/dL)": "MGlucose"})
                                # converts timestamp to datetime 
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # replaces historic glucose with scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # resamples to 5 minute intervals to have unifrom sample rate
                                hupa_glc = df_resample(hupa_glc, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM")                                                            

                                # adds the glucose file to a list of glucose files
                                all_data_GLC_h.append(hupa_glc)
                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}") 
                    # if file contains dexcom and is a csv file, the file is read; these contain CGM measurements every 5 minutes              
                    elif file.endswith(".csv") and "dexcom" in file:
                        # joins paths
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        try:
                            # dataframes with the CGM measurements are read with the "smart_read()" function
                            hupa_glc_d = smart_read(file_path_glc_h)
                            # adds Subject ID
                            hupa_glc_d["PtID"] = subject_id_h
                            # column names are renamed for semantic equality
                            hupa_glc_d = hupa_glc_d .rename(columns={"Marca temporal (AAAA-MM-DDThh:mm:ss)" : "ts", "Tipo de evento": "Type", "Nivel de glucosa (mg/dl)": "GlucoseCGM"})
                            # keeps only eventtype = Niveles estimados de glucosa
                            hupa_glc_d = hupa_glc_d [hupa_glc_d ["Type"] == "Niveles estimados de glucosa"][["ts", "PtID", "GlucoseCGM"]]
                            # timestamp is converted to datetime
                            hupa_glc_d["ts"] = pd.to_datetime(hupa_glc_d["ts"]) 
                            # adds dataframe to list of glucose dataframes
                            all_data_GLC_h.append(hupa_glc_d)

                        except Exception as e:
                            print(f"Failed to read {file_path_glc_h}: {e}")


        # concatenates all HR dataframes into one dataframe
        df_hupa_HR = pd.concat(all_data_HR_h, ignore_index=True)
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_hupa_HR = df_hupa_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))

        # concatenates all HR dataframes into one dataframe
        df_hupa_GLC = pd.concat(all_data_GLC_h, ignore_index=True)
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_hupa_GLC = df_hupa_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency = frequency, mode = "glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merges dataframes of HR and CGM data
        df_HUPA = pd.merge(df_hupa_GLC, df_hupa_HR, on=["PtID", "ts"], how="left")
        # reads dataframe with demographic information
        df_HUPA_info = smart_read("Data/datasets_for_T1D/demographics_hupa.csv")
        # computes the age of diabetes diagnosis
        df_HUPA_info["DiagAge"] = df_HUPA_info["Age"] - df_HUPA_info["DiabetesHistory"]

        # merges maindatabase with demographic information
        df_HUPA = pd.merge(df_HUPA, df_HUPA_info, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_HUPA["initial_date"] = df_HUPA.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_HUPA["Birthyear"] = df_HUPA["initial_date"].dt.year - df_HUPA["Age"] 
        # drop age column 
        df_HUPA = df_HUPA.drop("Age", axis=1)
        # computes the age of each year based on the birthyear
        df_HUPA["Age"] = df_HUPA["ts"].dt.year - df_HUPA["Birthyear"]

        df_HUPA = df_HUPA.rename(columns={"HbAc(%)": "Mean_Hba1c", "DiabetesHistory": "LOfDiabDiag"})

        # adds the database name to the patient ID to enable reidentification 
        df_HUPA["PtID"] = df_HUPA["PtID"] + "_HUPA-UCM"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_HUPA["Database"] = "HUPA-UCM"
        return df_HUPA
    

    def df_pedap():
        # This database contains two different CGM dataframes for some subjects, both dataframe with the CGM measurements are read with the "smart_read()" function
        df_pedap= smart_read("Data/datasets_for_T1D/PEDAP/Data Files/PEDAPDexcomClarityCGM.txt")
        # reduces the columns to only important columns
        df_pedap = df_pedap[["PtID", "RecID", "DeviceDtTm", "CGM"]]

        df_pedap_other = smart_read("Data/datasets_for_T1D/PEDAP/Data Files/PEDAPOtherCGM.txt")
        # reduces the columns to only important columns
        df_pedap_other = df_pedap_other[["PtID", "RecID", "DeviceDtTm", "CGM"]]
        # concatenates both dataframes into one dataframes
        df_PEDAP = pd.concat([df_pedap, df_pedap_other])


        # converts timstamps to datetime
        df_PEDAP["ts"] = pd.to_datetime(df_PEDAP["DeviceDtTm"], format = "mixed")
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_PEDAP = df_PEDAP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "CGM"))

        # reads dataframe of sex data
        df_PEDAP_screen = smart_read("Data/datasets_for_T1D/PEDAP/Data Files/PEDAPDiabScreening.txt")
        # reduces the columns to only important columns
        df_PEDAP_screen = df_PEDAP_screen[["PtID", "Sex", "Ethnicity", "Race", "DiagDt", "DiagAge", "Weight", "WeightUnits", "Height", "HeightUnits"]]


        # reads dataframe of age data
        df_PEDAP_age = smart_read("Data/datasets_for_T1D/PEDAP/Data Files/PtRoster.txt")
        # reduces the columns to only important columns
        df_PEDAP_age = df_PEDAP_age[["PtID", "AgeAsofEnrollDt"]]

        # merges dataframes including age and sex
        df_PEDAP_screen = pd.merge(df_PEDAP_screen, df_PEDAP_age, on="PtID", how="left")
        # merges dataframes of demographics and CGM data
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_screen, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_PEDAP["initial_date"] = df_PEDAP.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_PEDAP["Birthyear"] = df_PEDAP["initial_date"].dt.year - df_PEDAP["AgeAsofEnrollDt"] 
        # computes the age of each year based on the birthyear
        df_PEDAP["Age"] = df_PEDAP["ts"].dt.year - df_PEDAP["Birthyear"]

        df_PEDAP["LOfDiabDiag"] = df_PEDAP["Age"] - df_PEDAP["DiagAge"]

        # reads the dataframe with the hbac1 results
        df_PEDAP_hbc = smart_read("Data/datasets_for_T1D/PEDAP/Data Files/PEDAPScreeningCTV.txt")
        # keeps only important data
        df_PEDAP_hbc = df_PEDAP_hbc[["PtID", "HbA1cTestDt", "HbA1cTestRes"]]
        # extracts the date of the measurement timestamp
        df_PEDAP_hbc["date"] = pd.to_datetime(pd.to_datetime(df_PEDAP_hbc["HbA1cTestDt"]).dt.date)
        # extracts only the date of the timestamp in the original database as a new column
        df_PEDAP["date"] = pd.to_datetime((df_PEDAP["ts"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_hbc, on=["PtID", "date"], how="left")


        # reads dataframes of the physical examination 
        df_PEDAP_physic = smart_read("Data/datasets_for_T1D/DCLP3/Data Files/DiabPhysExam_a.txt")
        # convert weight to kg for semantic equality
        df_PEDAP_physic["WeightKg"] = df_PEDAP_physic["Weight"]
        df_PEDAP_physic.loc[df_PEDAP_physic["WeightUnits"] == "lbs", "WeightKg"] = df_PEDAP_physic.loc[df_PEDAP_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237
        # convert height to cm for semantic equality
        df_PEDAP_physic["HeightCm"] = df_PEDAP_physic["Height"]
        df_PEDAP_physic.loc[df_PEDAP_physic["HeightUnits"] == "in", "HeightCm"] = df_PEDAP_physic.loc[df_PEDAP_physic["HeightUnits"] == "in", "Height"] * 2.54
        # keeps only important columns
        df_PEDAP_physic = df_PEDAP_physic[["PtID", "WeightKg", "HeightCm","BldPrSys","BldPrDia", "PEHeartRt", "Temp", "TempUnits", "FingStkBG", "FingStkBGUnits"]]
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_physic, on=["PtID"], how="left")


        # reads lab based Hba1c results
        df_PEDAP_hbac1_sample = smart_read("Data/datasets_for_T1D/PEDAP/Data Files/STASampleResults.txt")
        # keeps only important data
        df_PEDAP_hbac1_sample = df_PEDAP_hbac1_sample[df_PEDAP_hbac1_sample["ResultName"] == "GLYHB"]
        df_PEDAP_hbac1_sample = df_PEDAP_hbac1_sample[["PtID", "Value", "AnalysisDt"]]
        # renames values 
        df_PEDAP_hbac1_sample = df_PEDAP_hbac1_sample.rename(columns={"Value": "LabHba1c"})
        df_PEDAP_hbac1_sample["date"] = pd.to_datetime(pd.to_datetime(df_PEDAP_hbac1_sample["AnalysisDt"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_hbac1_sample, on=["PtID", "date"], how="left")
        #df_PEDAP["HbA1cTestRes"] = df_PEDAP["HbA1cTestRes"].fillna(df_PEDAP["LabHba1c"])

        # column names are renamed for semantic equality
        df_PEDAP = df_PEDAP.rename(columns={"CGM": "GlucoseCGM", "HbA1cTestRes" : "Hba1c", "AnalysisDt": "LabAnalysisDt"})
        # adds the database name to the patient ID to enable reidentification 
        df_PEDAP["PtID"] = df_PEDAP["PtID"].astype(str) + "_PEDAP"
        # reduces maindatabase to only important values
        df_PEDAP = df_PEDAP[["ts", "PtID", "DeviceDtTm", "GlucoseCGM", "Sex", "Ethnicity",
       "Race", "DiagDt", "DiagAge", "Age",
       "LOfDiabDiag",  "HbA1cTestDt", "Hba1c", "WeightKg", "HeightCm",
     "LabHba1c", "LabAnalysisDt"]]
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_PEDAP["Database"] = "PEDAP"
        return df_PEDAP

    def df_replace():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_RBG = smart_read("Data/datasets_for_T1D/ReplaceBG/Data Tables/HDeviceCGM.txt")

        # column names are renamed for semantic equality
        df_RBG = df_RBG.rename(columns={"GlucoseValue": "GlucoseCGM"})
        # splits CGM and finger prick glucose levels into separate columns
        df_RBG["mGLC"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "Calibration")
        df_RBG["GlucoseCGM"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "CGM")

        # initial date is set
        df_RBG["initdate"] = pd.to_datetime("2024-01-01")
        # time is added to the data 
        df_RBG["datetime"] = df_RBG["initdate"] + pd.to_timedelta(df_RBG["DeviceDtTmDaysFromEnroll"], unit="D")
        df_RBG["DeviceTm"] = df_RBG["DeviceTm"].astype(str)

        # converts date and time to datetime and combines them into one column
        df_RBG["ts"] = pd.to_datetime(df_RBG["datetime"].dt.strftime("%Y-%m-%d") + " " + df_RBG["DeviceTm"])

        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_RBG = df_RBG.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe with sex data
        df_RBG_screen = smart_read("Data/datasets_for_T1D/REPLACEBG/Data Tables/HScreening.txt")
        # reduces the columns to only important columns
        df_RBG_screen = df_RBG_screen[["PtID", "Gender", "Ethnicity", "Race", "DiagAge", "Weight", "Height"]]

        # reads dataframe with age data
        df_RBG_age = smart_read("Data/datasets_for_T1D/REPLACEBG/Data Tables/HPtRoster.txt")
        # reduces the columns to only important columns
        df_RBG_age = df_RBG_age[["PtID", "AgeAsOfEnrollDt"]]

        # merges dataframes of sex and age data
        df_RBG_screen = pd.merge(df_RBG_screen, df_RBG_age, on="PtID", how="left")

        # merges dataframes of demographics and CGM data 
        df_RBG = pd.merge(df_RBG, df_RBG_screen, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_RBG["initial_date"] = df_RBG.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_RBG["Birthyear"] = df_RBG ["initial_date"].dt.year - df_RBG["AgeAsOfEnrollDt"] 
        # computes the age of each year based on the birthyear
        df_RBG["Age"] = df_RBG ["ts"].dt.year - df_RBG["Birthyear"]

        df_RBG["LOfDiabDiag"] = df_RBG["Age"] - df_RBG["DiagAge"]

        df_RBG_hbc = smart_read("Data/datasets_for_T1D/REPLACEBG/Data Tables/HLocalHbA1c.txt")
        df_RBG_hbc["initdate"] = pd.to_datetime("2024-01-01")
        df_RBG_hbc["datetime"] = df_RBG_hbc["initdate"] + pd.to_timedelta(df_RBG_hbc["HbA1cTestDtDaysAfterEnroll"], unit="D")
        df_RBG_hbc["date"] = pd.to_datetime(pd.to_datetime(df_RBG_hbc["datetime"]).dt.date)
        df_RBG_hbc = df_RBG_hbc[["PtID", "HbA1cTestDtDaysAfterEnroll", "HbA1cTestRes", "date"]] 

        df_RBG["date"] = pd.to_datetime((df_RBG["ts"]).dt.date)
        df_RBG = pd.merge(df_RBG, df_RBG_hbc, on=["PtID", "date"], how="left")


        # reads lab based Hba1c results
        df_RBG_lab = smart_read("Data/datasets_for_T1D/REPLACEBG/Data Tables/Sample.txt")
        # keeps only important data
        df_RBG_lab = df_RBG_lab[df_RBG_lab["Analyte"] == "HBA1C"]
        df_RBG_lab = df_RBG_lab[["PtID", "CollectionDtDaysFromEnroll", "Analyte", "Value"]]
        # renames values 
        df_RBG_lab = df_RBG_lab.rename(columns={"Value": "LabHba1c"})

        df_RBG_lab["initdate"] = pd.to_datetime("2024-01-01")
        # time is added to the data 
        df_RBG_lab["date"] = df_RBG_lab["initdate"] + pd.to_timedelta(df_RBG_lab["CollectionDtDaysFromEnroll"], unit="D")

        # combines the HbA1cTestResults with the main database
        df_RBG = pd.merge(df_RBG, df_RBG_lab, on=["PtID", "date"], how="left")
        df_RBG["HbA1cTestRes"] = df_RBG["HbA1cTestRes"].fillna(df_RBG["LabHba1c"])


        # column names are renamed for semantic equality
        df_RBG = df_RBG.rename(columns={"Gender": "Sex", "HbA1cTestRes": "Hba1c", "Height": "HeightCm", "Weight": "WeightKg"}) 

        df_RBG = df_RBG[["ts", "PtID",
            "DexInternalDtTmDaysFromEnroll", "DexInternalTm", 
            "GlucoseCGM", "mGLC", "Sex", "Ethnicity",
            "Race", "DiagAge", "WeightKg", "HeightCm",
            "Age", "LOfDiabDiag", "Hba1c", "LabHba1c"]]

        # adds the database name to the patient ID to enable reidentification 
        df_RBG["PtID"] = df_RBG["PtID"].astype(str) + "_RBG"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_RBG["Database"] = "RBG"
        return df_RBG

    def df_sence():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_SENCE = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/DeviceCGM.txt")

        # converts timstamps to datetime
        df_SENCE["ts"] = pd.to_datetime(df_SENCE["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_SENCE = df_SENCE.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "Value"))

        # reads dataframe with sex data
        df_SENCE_screen = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_SENCE_screen = df_SENCE_screen[["PtID", "Gender", "Ethnicity", "Race", "DiagDt",  "DiagAge"]]

        # reads dataframe with age data
        df_SENCE_age = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_SENCE_age = df_SENCE_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merges dataframes of age and sex data   
        df_SENCE_screen = pd.merge(df_SENCE_screen, df_SENCE_age, on="PtID", how="left")

        # merges dataframes of demographics with CGM data
        df_SENCE = pd.merge(df_SENCE, df_SENCE_screen, on=["PtID"], how="left")


        # extracts the date of initial measurement
        df_SENCE["initial_date"] = df_SENCE.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_SENCE["Birthyear"] = df_SENCE["initial_date"].dt.year - df_SENCE["AgeAsOfEnrollDt"] 
        # computes the age of each year based on the birthyear
        df_SENCE["Age"] = df_SENCE["ts"].dt.year - df_SENCE["Birthyear"]

        df_SENCE["LOfDiabDiag"] = df_SENCE["Age"] - df_SENCE["DiagAge"]

        # reads the dataframe with the hbac1 results
        df_SENCE_hbc = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/DiabLocalHbA1c.txt")
        # keeps only important data
        df_SENCE_hbc = df_SENCE_hbc[["PtID", "HbA1cTestDt", "HbA1cTestRes"]]
        # extracts the date of the measurement timestamp
        df_SENCE_hbc["date"] = pd.to_datetime(pd.to_datetime(df_SENCE_hbc["HbA1cTestDt"]).dt.date)
        # extracts only the date of the timestamp in the original database as a new column
        df_SENCE["date"] = pd.to_datetime((df_SENCE["ts"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_SENCE = pd.merge(df_SENCE, df_SENCE_hbc, on=["PtID", "date"], how="left")

        # reads lab based Hba1c results
        df_SENCE_hbac1_sample = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/STASampleResults.txt")
        # keeps only important data
        df_SENCE_hbac1_sample = df_SENCE_hbac1_sample[df_SENCE_hbac1_sample["ResultName"] == "GLYHB"]
        df_SENCE_hbac1_sample = df_SENCE_hbac1_sample[["PtID", "Value", "CollectionDt"]]
        # renames values 
        df_SENCE_hbac1_sample = df_SENCE_hbac1_sample.rename(columns={"Value": "LabHba1c"})
        df_SENCE_hbac1_sample["date"] = pd.to_datetime(pd.to_datetime(df_SENCE_hbac1_sample["CollectionDt"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_SENCE = pd.merge(df_SENCE, df_SENCE_hbac1_sample, on=["PtID", "date"], how="left")
        df_SENCE["HbA1cTestRes"] = df_SENCE["HbA1cTestRes"].fillna(df_SENCE["LabHba1c"])


        # reads dataframes of the physical examination 
        df_SENCE_physic = smart_read("Data/datasets_for_T1D/SENCE/Data Tables/DiabPhysExam.txt")
        # keeps only important columns
        df_SENCE_physic = df_SENCE_physic[["PtID", "Visit", 
            "Weight", "WeightUnits", "Height", "HeightUnits", "BldPrSys", "BldPrDia", "BldPrUnk", "PEHeartRt",
                "Temp", "TempUnits", "FingStkBG",
            "FingStkBGUnits"]]
        # convert weight to kg for semantic equality
        df_SENCE_physic["WeightKg"] = df_SENCE_physic["Weight"]
        df_SENCE_physic.loc[df_SENCE_physic["WeightUnits"] == "lbs", "WeightKg"] = df_SENCE_physic.loc[df_SENCE_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237
        # convert height to cm for semantic equality
        df_SENCE_physic["HeightCm"] = df_SENCE_physic["Height"]
        df_SENCE_physic.loc[df_SENCE_physic["HeightUnits"] == "in", "HeightCm"] = df_SENCE_physic.loc[df_SENCE_physic["HeightUnits"] == "in", "Height"] * 2.54
        df_SENCE = pd.merge(df_SENCE, df_SENCE_physic, on=["PtID"], how="left")


        # column names are renamed for semantic equality
        df_SENCE = df_SENCE.rename(columns={"Value": "GlucoseCGM", "Gender": "Sex","HbA1cTestRes": "Hba1c"})
        # adds the database name to the patient ID to enable reidentification 
        df_SENCE["PtID"] = df_SENCE["PtID"].astype(str) + "_SENCE"

        df_SENCE = df_SENCE[["ts", "PtID",  "GlucoseCGM", "Sex", "Ethnicity", "Race",
            "DiagAge", "LOfDiabDiag",  "Birthyear", "Age", "HbA1cTestDt", "Hba1c",
            "LabHba1c", "Visit", "WeightKg", "HeightCm"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_SENCE["Database"] = "SENCE"
        return df_SENCE

    def df_shd():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_SHD = smart_read("Data/datasets_for_T1D/SevereHypoDataset/Data Tables/BDataCGM.txt")

        # initial date is set
        df_SHD["initdate"] = pd.to_datetime("2023-01-01")
        # adds date with time and converts into datetime
        df_SHD["datetime"] = df_SHD["initdate"] + pd.to_timedelta(df_SHD["DeviceDaysFromEnroll"], unit="D")
        df_SHD["DeviceTm"] = df_SHD["DeviceTm"].astype(str)

        # converts timstamps to datetime
        df_SHD["ts"] = pd.to_datetime(df_SHD["datetime"].dt.strftime("%Y-%m-%d") + " " + df_SHD["DeviceTm"])

        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_SHD = df_SHD.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "Glucose"))

        # reads dataframe with sex data
        df_SHD_screen = smart_read("Data/datasets_for_T1D/SevereHypoDataset/Data Tables/BDemoLifeDiabHxMgmt.txt")
        # reduces the columns to only important columns
        df_SHD_screen = df_SHD_screen[["PtID", "Gender", "Ethnicity", "Race", "T1DDiagAge"]]
        # true age is not given but it is told that patients are aged at least 60
        df_SHD_screen["Age"] = "60-100" 

        # merges dataframe of demographics with CGM data
        df_SHD = pd.merge(df_SHD, df_SHD_screen, on="PtID", how="left")


        df_SHD_physic = smart_read("Data/datasets_for_T1D/SevereHypoDataset/Data Tables/BMedChart.txt")
        df_SHD_physic = df_SHD_physic[["PtID", "VisitDaysFromEnroll", "Weight", "WeightUnits", "Height", "HeightUnits"]]

        # convert weight to kg for semantic equality
        df_SHD_physic["WeightKg"] = df_SHD_physic["Weight"]
        df_SHD_physic.loc[df_SHD_physic["WeightUnits"] == "lbs", "WeightKg"] = df_SHD_physic.loc[df_SHD_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237
        # convert height to cm for semantic equality
        df_SHD_physic["HeightCm"] = df_SHD_physic["Height"]
        df_SHD_physic.loc[df_SHD_physic["HeightUnits"] == "in", "HeightCm"] = df_SHD_physic.loc[df_SHD_physic["HeightUnits"] == "in", "Height"] * 2.54

        # initial date is set
        df_SHD_physic["initdate"] = pd.to_datetime("2023-01-01")
        # adds date with time and converts into datetime
        df_SHD_physic["date"] = df_SHD_physic["initdate"] + pd.to_timedelta(df_SHD_physic["VisitDaysFromEnroll"], unit="D")

        # extracts only the date of the timestamp in the original database as a new column
        df_SHD["date"] = pd.to_datetime((df_SHD["ts"]).dt.date)
        # merges physics with main database
        df_SHD = pd.merge(df_SHD, df_SHD_physic, on=["PtID", "date"], how="left")


        # reads the dataframe with the hbac1 results
        df_SHD_hbc = smart_read("Data/datasets_for_T1D/SevereHypoDataset/Data Tables/BSampleResults.txt")
        df_SHD_hbc = df_SHD_hbc[df_SHD_hbc["ResultName"] == "HbA1c"]
        df_SHD_hbc["DeviceDaysFromEnroll"] = 0
        # keeps only important data
        df_SHD_hbc = df_SHD_hbc[["PtID", "Value", "DeviceDaysFromEnroll"]]
        # renames values 
        df_SHD_hbc = df_SHD_hbc.rename(columns={"Value": "Hba1c"})
        # combines the HbA1cTestResults with the main database
        df_SHD = pd.merge(df_SHD, df_SHD_hbc, on=["PtID", "DeviceDaysFromEnroll"], how="left")



        # column names are renamed for semantic equality
        df_SHD = df_SHD.rename(columns={"Glucose": "GlucoseCGM", "Gender": "Sex", "T1DDiagAge": "DiagAge"})
        # adds the database name to the patient ID to enable reidentification 
        df_SHD["PtID"] = df_SHD["PtID"].astype(str) + "_SHD"

        df_SHD = df_SHD[["ts", "PtID",
            "DeviceDaysFromEnroll", "DeviceTm", "GlucoseCGM", "Sex", "Ethnicity", "Race", "DiagAge", "Age", 
            "WeightKg", "HeightCm", "Hba1c"]]


        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_SHD["Database"] = "SHD"
        return df_SHD

    def df_wisdm():
        # this dataset has two different CGM records which are read seperately with the "smart_read()" function
        df_WISDM_device = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/DeviceCGM.txt") 
        # reduces the columns to only important columns
        df_WISDM_device = df_WISDM_device[["PtID", "DeviceDtTm", "Value"]]

        df_WISD_ext = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/cgmAnalysis Ext.txt")
        # reduces the columns to only important columns
        df_WISD_ext = df_WISD_ext[["PtID", "DeviceDtTm", "Value"]]
        df_WISD_ext["DeviceDtTm"] = pd.to_datetime(df_WISD_ext["DeviceDtTm"], format="%d%b%Y:%H:%M:%S.%f")

        # merges all dataframes
        df_WISDM = pd.concat([df_WISDM_device, df_WISD_ext])


        # converts timstamps to datetime
        df_WISDM["ts"] = pd.to_datetime(df_WISDM["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_WISDM = df_WISDM.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "Value"))

        # reads dataframe with sex data
        df_WISDM_screen = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_WISDM_screen = df_WISDM_screen[["PtID","Gender", "Ethnicity", "Race", "DiagDt", "DiagDtApprox", 
            "DiagAge", "DiagAgeApprox"]]

        # reads dataframe with age data
        df_WISDM_age = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_WISDM_age = df_WISDM_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merge dataframes of sex and age
        df_WISDM_screen = pd.merge(df_WISDM_screen, df_WISDM_age, on="PtID", how="left")

        # merges dataframes of demographics with CGM data
        df_WISDM = pd.merge(df_WISDM, df_WISDM_screen, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_WISDM["initial_date"] = df_WISDM.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_WISDM["Birthyear"] = df_WISDM["initial_date"].dt.year - df_WISDM["AgeAsOfEnrollDt"] 
        # computes the age of each year based on the birthyear
        df_WISDM["Age"] = df_WISDM["ts"].dt.year - df_WISDM["Birthyear"]

        df_WISDM["LOfDiabDiag"] = df_WISDM["Age"] - df_WISDM["DiagAge"]


        # reads dataframe with age data
        df_WISDM_physic = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/DiabPhysExam.txt")
        # keeps only important columns
        df_WISDM_physic = df_WISDM_physic[["PtID", "Weight", "WeightUnits", "Height", "HeightUnits",
            "BldPrSys", "BldPrDia", "PEHeartRt", "Temp", "TempUnits", "FingStkBG", "FingStkBGUnits"]]
        # convert weight to kg for semantic equality
        df_WISDM_physic["WeightKg"] = df_WISDM_physic["Weight"]
        df_WISDM_physic.loc[df_WISDM_physic["WeightUnits"] == "lbs", "WeightKg"] = df_WISDM_physic.loc[df_WISDM_physic["WeightUnits"] == "lbs", "Weight"] * 0.45359237
        # convert height to cm for semantic equality
        df_WISDM_physic["HeightCm"] = df_WISDM_physic["Height"]
        df_WISDM_physic.loc[df_WISDM_physic["HeightUnits"] == "in", "HeightCm"] = df_WISDM_physic.loc[df_WISDM_physic["HeightUnits"] == "in", "Height"] * 2.54
        # merge demographics with main database
        df_WISDM = pd.merge(df_WISDM, df_WISDM_physic, on=["PtID"], how="left")


        # reads the dataframe with the hbac1 results
        df_WISDM_hbc = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/DiabLocalHbA1c.txt")
        # keeps only important data
        df_WISDM_hbc = df_WISDM_hbc[["PtID", "HbA1cTestDt", "HbA1cTestRes"]]
        # extracts the date of the measurement timestamp
        df_WISDM_hbc["date"] = pd.to_datetime(pd.to_datetime(df_WISDM_hbc["HbA1cTestDt"]).dt.date)
        # extracts only the date of the timestamp in the original database as a new column
        df_WISDM["date"] = pd.to_datetime((df_WISDM["ts"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_WISDM = pd.merge(df_WISDM, df_WISDM_hbc, on=["PtID", "date"], how="left")

        # reads lab based Hba1c results
        df_WISDM_hbac1_sample = smart_read("Data/datasets_for_T1D/WISDM/Data Tables/STASampleResults.txt")
        # keeps only important data
        df_WISDM_hbac1_sample = df_WISDM_hbac1_sample[df_WISDM_hbac1_sample["ResultName"] == "GLYHB"]
        df_WISDM_hbac1_sample = df_WISDM_hbac1_sample[["PtID", "Value", "CollectionDt"]]
        # renames values 
        df_WISDM_hbac1_sample = df_WISDM_hbac1_sample.rename(columns={"Value": "LabHba1c"})
        df_WISDM_hbac1_sample["date"] = pd.to_datetime(pd.to_datetime(df_WISDM_hbac1_sample["CollectionDt"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_WISDM = pd.merge(df_WISDM, df_WISDM_hbac1_sample, on=["PtID", "date"], how="left")
        df_WISDM["HbA1cTestRes"] = df_WISDM["HbA1cTestRes"].fillna(df_WISDM["LabHba1c"])


        # column names are renamed for semantic equality
        df_WISDM = df_WISDM.rename(columns={"Value": "GlucoseCGM", "Gender": "Sex", "HbA1cTestRes": "Hba1c"})
        # adds the database name to the patient ID to enable reidentification 
        df_WISDM["PtID"] = df_WISDM["PtID"].astype(str) + "_WISDM"

        df_WISDM = df_WISDM[["ts", "PtID", "DeviceDtTm", "GlucoseCGM", "Sex", "Ethnicity", "Race",
            "DiagAge", "DiagAgeApprox", "LOfDiabDiag",
            "Age", "WeightKg", "HeightCm", "HbA1cTestDt", "Hba1c", "LabHba1c", "CollectionDt"]]


        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_WISDM["Database"] = "WISDM"
        return df_WISDM

    def df_shanghai():
        # path to the excel files
        file_paths_shang = glob.glob("Data/datasets_for_T1D/shanghai/Shanghai_T1DM/*.xlsx")  # Change path accordingly

        # initializes an empty list to store dataframes
        df_list_shang = []

        # loops through each file
        for idx, file in enumerate(file_paths_shang, start=1):
            # reads the dataframes containing CGM measurements with the "smart_read()" function
            df = smart_read(file) 
            # extracts the Subject IDs from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # assings the unique PtID
            df["PtID"] = str(name)  
            # adds the dataframe to a list
            df_list_shang.append(df)
            
        # concatenates all dataframes into one dataframe
        df_shang = pd.concat(df_list_shang, ignore_index=True)

        # path to the second set of files with the xls extension
        file_paths_shang_2 = glob.glob("Data/datasets_for_T1D/shanghai/Shanghai_T1DM/*.xls")  # Change path accordingly

        # initializes an empty list to store dataframes
        df_list_shang_2  = []
        idxx = 5
        # loops through each file
        for idx, file in enumerate(file_paths_shang_2 , start=1):
            # reads the dataframes containing CGM measurements with the "smart_read()" function
            df = smart_read(file)  
            # extracts the Subject IDs from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # assings the unique PtID
            df["PtID"] = str(name) 
            # adds the dataframe to a list
            df_list_shang_2 .append(df)
            idxx = idxx + 1
            
        # concatenates all dataframes into one dataframe
        df_shang_2  = pd.concat(df_list_shang_2, ignore_index=True)
        df_shang  = pd.concat([df_shang, df_shang_2], ignore_index=True)
        df_shang = df_shang[["Date", "CGM (mg / dl)",
            "Dietary intake", "Insulin dose - s.c.",
            "CSII - bolus insulin (Novolin R, IU)",
            "CSII - basal insulin (Novolin R, IU / H)", "Insulin dose - i.v.",
            "PtID"]]

        # converts timstamps to datetime
        df_shang["ts"] = pd.to_datetime(df_shang["Date"])
        # undersamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_shang  = df_shang.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "CGM (mg / dl)"))                                                                                   

        # reads the dataframe with demographic data
        df_shang_info = smart_read("Data/datasets_for_T1D/shanghai/Shanghai_T1DM_Summary.xlsx")
        # reduces the columns to only important columns
        df_shang_info = df_shang_info[["Patient Number", "Gender (Female=1, Male=2)", "Age (years)", "Height (m)", "Weight (kg)", "Duration of Diabetes  (years)", "HbA1c (mmol/mol)"]]
        # column names are renamed for semantic equality
        df_shang_info["Sex"] = df_shang_info["Gender (Female=1, Male=2)"].replace({2: "M", 1: "F"})
        # height is measured in meters so it need to be converted to cm 
        df_shang_info["Height (m)"] = df_shang_info["Height (m)"] * 100 
        # column names are renamed for semantic equality
        df_shang_info = df_shang_info.rename(columns={"Patient Number": "PtID", "Height (m)": "HeightCm", "Weight (kg)": "WeightKg"})

        # copies dataset for later use of Hba1c value
        shang_temp_hba1c = df_shang_info.copy()
        # drops hba1c value from dataset
        df_shang_info = df_shang_info.drop("HbA1c (mmol/mol)", axis=1)

        # merges dataframes with demographics and CGM data
        df_shang = pd.merge(df_shang, df_shang_info, on=["PtID"], how="left")


        # extracts the date of initial measurement
        df_shang["initial_date"] = df_shang.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_shang["Birthyear"] = df_shang["initial_date"].dt.year - df_shang["Age (years)"] 
        # computes the age of each year based on the birthyear
        df_shang["Age"] = df_shang["ts"].dt.year - df_shang["Birthyear"]

        # calculate age of diabetes diagnosis for semantic equality
        df_shang["DiagAge"] =  df_shang["Age"] - df_shang["Duration of Diabetes  (years)"]
        df_shang["initial_date_only"] = df_shang["ts"].dt.date.astype(str) 


        shang_temp_hba1c = shang_temp_hba1c[["PtID","HbA1c (mmol/mol)" ]]
        # converts to numeric values to be safe
        shang_temp_hba1c["HbA1c (mmol/mol)"] = pd.to_numeric(shang_temp_hba1c["HbA1c (mmol/mol)"], errors="coerce")
        # calculate Hba1c in percentage for semantic equality
        shang_temp_hba1c["Hba1c"] = (shang_temp_hba1c["HbA1c (mmol/mol)"]/10.929) +  + 2.15 
        first_dates = df_shang.groupby("PtID")["ts"].min().reset_index()
        first_dates.rename(columns={"ts": "initial_date_only"}, inplace=True)

        # merges first_dates with PtIds
        shang_temp_hba1c = shang_temp_hba1c.merge(first_dates, on="PtID", how="left")
        shang_temp_hba1c["initial_date_only"] = shang_temp_hba1c["initial_date_only"].dt.date.astype(str)
        shang_temp_hba1c = shang_temp_hba1c.dropna(subset=["Hba1c"])

        df_shang = pd.merge(df_shang, shang_temp_hba1c, on=["PtID", "initial_date_only"], how="left")


        # column names are renamed for semantic equality
        df_shang = df_shang.rename(columns={"CGM (mg / dl)": "GlucoseCGM", "Duration of Diabetes  (years)": "LOfDiabDiag"})
        # adds the database name to the patient ID to enable reidentification 
        df_shang["PtID"] = df_shang["PtID"] + "_ShanghaiT1D" 

        # reduces database to important values
        df_shang = df_shang[["ts", "Date", "GlucoseCGM",
            "PtID", "HeightCm",
            "WeightKg", "LOfDiabDiag", "Sex",
            "Hba1c", "Age", "DiagAge"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_shang["Database"] = "ShanghaiT1D"
        return df_shang

    def df_d1namo():
        # base path 
        base_path_D1namo_ECG = "Data/datasets_for_T1D/D1NAMO/diabetes_subset"

        # initiliazes list to store all dataframes
        all_data_HR = []
        all_data_GLC = []

        # loop through each subject directory 
        for subject_id in os.listdir(base_path_D1namo_ECG):

            # reads all files including CGM readings
            person_path_glc = os.path.join(base_path_D1namo_ECG, subject_id)

            # skips if not a directory
            if not os.path.isdir(person_path_glc):
                continue

            # loops through each file of the subject
            for file in os.listdir(person_path_glc):
                
                # reads the file if file is a csv and ends with glucose
                if file.endswith("glucose.csv"):
                    # joins paths 
                    file_path_glc = os.path.join(person_path_glc, file)

                    try:
                        # reads the dataframes with the CGM measurements
                        D1NAMO_glc = smart_read(file_path_glc)
                        # assigns a Subject ID
                        D1NAMO_glc["PtID"] = subject_id
                        # column names are renamed for semantic equality
                        D1NAMO_glc = D1NAMO_glc.rename(columns={"glucose": "GlucoseCGM"})
                        # converts mmol/L to mg/dL
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"] * 18.02
                        # converts timstamps to datetime
                        D1NAMO_glc["ts"] = pd.to_datetime(D1NAMO_glc["date"] + " " + D1NAMO_glc["time"])

                        # splits manual and continuous glucose data into seperate columns
                        D1NAMO_glc["mGLC"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "manual")
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "cgm")
                        # add the single dataframes to the list
                        all_data_GLC.append(D1NAMO_glc)

                    except Exception as e:
                        print(f"Failed to read {file_path_glc}: {e}")

            # read the dataframes including heartrate measurements
            person_path = os.path.join(base_path_D1namo_ECG, subject_id, "sensor_data")

            if os.path.isdir(person_path):

                for session_folder in os.listdir(person_path):
                    session_path = os.path.join(person_path, session_folder)

                    # skips if not a directory
                    if not os.path.isdir(session_path):
                        continue
                    # loops through each file
                    for file in os.listdir(session_path):
                        # reads the file if it ends with _Summary.csv
                        if file.endswith("_Summary.csv"): 
                            # joins the paths
                            file_path = os.path.join(session_path, file)
                            
                            try:
                                # reads the dataframes with the heartrate values
                                D1NAMO_hr = smart_read(file_path)
                                # assigns the Subject ID 
                                D1NAMO_hr["PtID"] = subject_id
                                # adds the single dataframes to the list 
                                all_data_HR.append(D1NAMO_hr)
                            except Exception as e:
                                print(f"Failed to read {file_path}: {e}")

        # concatenates all HR dataframes into one dataframe
        df_D1NAMO_HR = pd.concat(all_data_HR, ignore_index=True)
        # concatenates all glucose dataframes into one dataframe
        df_D1NAMO_GLC = pd.concat(all_data_GLC, ignore_index=True)

        # reduces columns to only important columns
        df_D1NAMO_HR = df_D1NAMO_HR[["Time", "PtID", "HR"]]
        # converts timstamps to datetime
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["Time"], format="%d/%m/%Y %H:%M:%S.%f")
        # the datetime needs to be converted to the same format 
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_D1NAMO_HR = df_D1NAMO_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_D1NAMO_GLC = df_D1NAMO_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merge dataframes of HR and CGM data
        df_D1NAMO = pd.merge(df_D1NAMO_GLC, df_D1NAMO_HR, on=["PtID", "ts"], how="left")
        # adds the database name to the patient ID to enable reidentification 
        df_D1NAMO["PtID"] = df_D1NAMO["PtID"] + "_D1NAMO" 
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_D1NAMO["Database"] = "D1NAMO"
        return df_D1NAMO

    def df_DDATSHR():
        # subjects wear the metronic or abbott CGM device, so both files are read separately
        # reads the dataframe with the CGM measurements with the "smart_read()" function 
        #  medtronic estimates glucose every 5 minutes      
        df_DDATSHR_ins = smart_read("Data/datasets_for_T1D/DDATSHR/data-csv/Medtronic.csv")
        df_DDATSHR_ins = df_DDATSHR_ins[["Subject code number", "Local datetime [ISO8601]",
            "Local date [yyyy-mm-dd]", "Local time [hh:mm:ss]", "UTC offset [hr]",
            "File", "Index", "New Device Time", "Sensor Glucose [mmol/l]"]]

        # abbott estiamtes glucose evry 15 minutes
        df_DDATSHR_glc = smart_read("Data/datasets_for_T1D/DDATSHR/data-csv/Abbott.csv")

        # converts timstamps to datetime
        df_DDATSHR_glc["ts"] = pd.to_datetime(df_DDATSHR_glc["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_glc["Local time [hh:mm]"])
        # for each subject, replaces the scan values with the historic glucose values
        df_DDATSHR_glc["Historic Glucose [mmol/l]"] = df_DDATSHR_glc.groupby("Subject code number").apply(
            lambda group: group["Scan Glucose [mmol/l]"].where(
                group["Scan Glucose [mmol/l]"].notna(), group["Historic Glucose [mmol/l]"]
            )
        ).reset_index(drop=True)
        # undersamples abbott data to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_glc  = df_DDATSHR_glc.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "Subject code number", value = "Historic Glucose [mmol/l]"))                                                                                   

        # removes all other timestamps which were used for insulin but have no glucose entry
        df_DDATSHR_ins = df_DDATSHR_ins.dropna(subset=["Sensor Glucose [mmol/l]"])

        # converts timstamps to datetime
        df_DDATSHR_ins["ts"] = pd.to_datetime(df_DDATSHR_ins["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_ins["Local time [hh:mm:ss]"])
        # column names are renamed for semantic equality
        df_DDATSHR_ins = df_DDATSHR_ins.rename(columns={"Sensor Glucose [mmol/l]": "Historic Glucose [mmol/l]"})
        # reduces the columns to only important columns
        df_DDATSHR_ins = df_DDATSHR_ins[["ts", "Subject code number", "Historic Glucose [mmol/l]"]]

        # merges dataframes of CGM measurements for both sensors into one dataframe
        df_DDATSHR_glc_ins = pd.concat([df_DDATSHR_glc, df_DDATSHR_ins])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_glc_ins = df_DDATSHR_glc_ins.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "Subject code number", value = "Historic Glucose [mmol/l]"))
        # converts CGM measured in mmol/L to mg/dL 
        df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] = df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] * 18.02


        #reads dataframes of heartrate data 
        df_DDATSHR_hr = smart_read("Data/datasets_for_T1D/DDATSHR/data-csv/Fitbit/Fitbit-heart-rate.csv")

        # converts timstamps to datetime
        df_DDATSHR_hr["ts"] = pd.to_datetime(df_DDATSHR_hr["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_hr["Local time [hh:mm]"])
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_hr = df_DDATSHR_hr.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "Subject code number", value = "heart rate [#/min]"))

        # reduces the columns to only important columns
        df_DDATSHR_hr = df_DDATSHR_hr[["ts", "Subject code number", "heart rate [#/min]"]]

        # merge dataframes of heartrate and CGM data 
        df_DDATSHR = pd.merge(df_DDATSHR_glc_ins, df_DDATSHR_hr, on=["Subject code number", "ts"], how="left")



        # reads dataframe of age and gender 
        df_DDATSHR_temp = smart_read("Data/datasets_for_T1D/DDATSHR/data-csv/population.csv")

        df_DDATSHR_info1 = df_DDATSHR_temp.copy()

        df_DDATSHR_info1 = df_DDATSHR_info1[["Subject code number", "Gender [M=male F=female]", "Length [cm]",
            "Weight [kg]", "Age [yr]", "HbA1C_1 [yyyy]", "HbA1C_1 [mm]",
            "HbA1C_1 [mmol/mol]"]]

        df_DDATSHR_info1["year"] = df_DDATSHR_info1["HbA1C_1 [yyyy]"].astype(float)
        df_DDATSHR_info1["month"] = df_DDATSHR_info1["HbA1C_1 [mm]"].astype(float)

        df_DDATSHR_info2 = df_DDATSHR_temp.copy()
        df_DDATSHR_info2 = df_DDATSHR_info2[["Subject code number","Gender [M=male F=female]", "Length [cm]",
            "Weight [kg]", "Age [yr]", "HbA1C_2 [yyyy]", "HbA1C_2 [mm]",
            "HbA1C_2 [mmol/mol]"]]

        df_DDATSHR_info2["year"] = df_DDATSHR_info2["HbA1C_2 [yyyy]"].astype(float)
        df_DDATSHR_info2["month"]  =  df_DDATSHR_info2["HbA1C_2 [mm]"].astype(float)


        df_DDATSHR_info2 = df_DDATSHR_info2.rename(columns={"HbA1C_2 [mmol/mol]":"HbA1C_1 [mmol/mol]"})

        df_DDATSHR_hba1c = pd.concat([df_DDATSHR_info1, df_DDATSHR_info2])
        # calculate Hba1c in percentage for semantic equality
        df_DDATSHR_hba1c["Hba1c"] = (df_DDATSHR_hba1c["HbA1C_1 [mmol/mol]"]/10.929) + 2.15 
        df_DDATSHR_hba1c = df_DDATSHR_hba1c.dropna(subset=["Hba1c"])

        df_DDATSHR_hba1c = df_DDATSHR_hba1c[["Subject code number", "year", "month","Hba1c"]]
        df_DDATSHR["year"] = df_DDATSHR["ts"].dt.year.astype(float)
        df_DDATSHR["month"] = df_DDATSHR["ts"].dt.month.astype(float)
        df_DDATSHR = pd.merge(df_DDATSHR, df_DDATSHR_hba1c, on=["Subject code number", "month", "year"], how="left")


        df_DDATSHR_info = df_DDATSHR_temp[["Subject code number","Gender [M=male F=female]", "Length [cm]", "Weight [kg]", "Age [yr]"]]
        # merges dataframes with demographics and CGM data
        df_DDATSHR = pd.merge(df_DDATSHR, df_DDATSHR_info, on=["Subject code number"], how="left")


        # extracts the date of initial measurement
        df_DDATSHR["initial_date"] = df_DDATSHR.groupby("Subject code number")["ts"].transform("min")
        # calculates the birthyear
        df_DDATSHR["Birthyear"] = df_DDATSHR["initial_date"].dt.year - df_DDATSHR["Age [yr]"] 
        # computes the age of each year based on the birthyear
        df_DDATSHR["Age"] = df_DDATSHR["ts"].dt.year - df_DDATSHR["Birthyear"]


        # column names are renamed for semantic equality
        df_DDATSHR = df_DDATSHR.rename(columns={"Subject code number":"PtID", "Historic Glucose [mmol/l]": "GlucoseCGM", "Gender [M=male F=female]" : "Sex", "heart rate [#/min]": "HR", "steps [#]": "Steps" , "Length [cm]": "HeightCm",
            "Weight [kg]": "WeightKg"})
        # adds the database name to the patient ID to enable reidentification 
        df_DDATSHR["PtID"] = df_DDATSHR["PtID"].astype(str) + "_DDATSHR"

        df_DDATSHR = df_DDATSHR[["ts", "PtID", "GlucoseCGM", "Sex", "HeightCm", "WeightKg",
            "Hba1c", "Birthyear", "Age", "HR"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_DDATSHR["Database"] = "DDATSHR"
        return df_DDATSHR
    
    def df_rtc():
        # path to the excel files 
        file_paths_rtc = glob.glob("Data/datasets_for_T1D/RT_CGM/DataTables/tblADataRTCGM*.csv") 

        # initializes an empty list to store dataframes
        df_list_rtc = []

        # loops through each file
        for idx, file in enumerate(file_paths_rtc, start=1):
            # reads the files with the "smart_read()" function
            df = smart_read(file)
            # adds the dataframe to the list
            df_list_rtc.append(df)

        # concatenates all dataframes into one dataframe
        df_rtc  = pd.concat(df_list_rtc , ignore_index=True)

        # converts timstamps to datetime depending on the format
        try: 
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"], format="%Y-%m-%d %H:%M:%S")

        # detects the sample rate since some subjects have glucose collected in 10 minute intervals; this is done separately for each subject
        fre_rtc = df_rtc.groupby("PtID", group_keys=False).apply(lambda x: detect_sample_rate(x, time_col = "ts")).reset_index(name="Frequency")
        # adds teh frequency column to the dataframe with CGM measurements
        df_RTC = df_rtc.merge(fre_rtc, on="PtID", how="left")

        # resamples the whole dataframe to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_RTC = df_RTC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "Glucose"))

        # reads dataframe with demographics
        df_rtc_info = smart_read("Data/datasets_for_T1D/RT_CGM/DataTables/tblAPtSummary.csv")
        # reduces the columns to only important columns
        df_rtc_info = df_rtc_info[["PtID", "Gender", "AgeAsOfRandDt", "Race", "Ethnicity", "Height", "Weight", "DavDiabetes"]]

        # merges dataframes of demographics with CGM data
        df_RTC = pd.merge(df_RTC, df_rtc_info, on=["PtID"], how="left")


        # reads the dataframe with the hbac1 results
        df_rtc_hbc = smart_read("Data/datasets_for_T1D/RT_CGM/DataTables/tblALabHbA1c.csv")
        # keeps only important data
        df_rtc_hbc =df_rtc_hbc[["PtID", "LabHbA1cDt", "LabA1cResult"]]
        # extracts the date of the measurement timestamp
        df_rtc_hbc["date"] = pd.to_datetime(pd.to_datetime(df_rtc_hbc["LabHbA1cDt"]).dt.date)
        # extracts only the date of the timestamp in the original database as a new column
        df_RTC["date"] = pd.to_datetime((df_RTC["ts"]).dt.date)
        # combines the HbA1cTestResults with the main database
        df_RTC  = pd.merge(df_RTC, df_rtc_hbc, on=["PtID", "date"], how="left")


        # extracts the date of initial measurement
        df_RTC["initial_date"] = df_RTC.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_RTC["Birthyear"] = df_RTC["initial_date"].dt.year - df_RTC["AgeAsOfRandDt"] 
        # computes the age of each year based on the birthyear
        df_RTC["Age"] = df_RTC["ts"].dt.year - df_RTC["Birthyear"]

        # column names are renamed for semantic equality
        df_RTC = df_RTC.rename(columns={"Glucose": "GlucoseCGM", "Gender": "Sex", "DavDiabetes":"DiagAge" , "LabA1cResult" : "Hba1c", "Height": "HeightCm", "Weight": "WeightKg"})
        df_RTC["LOfDiabDiag"] = df_RTC["Age"] - df_RTC["DiagAge"]

        # adds the database name to the patient ID to enable reidentification 
        df_RTC["PtID"] = df_RTC["PtID"].astype(str) + "_RT-CGM"

        df_RTC = df_RTC[["ts", "PtID", "DeviceDtTm", "GlucoseCGM", "Frequency", "Sex",
            "AgeAsOfRandDt", "Race", "Ethnicity", "HeightCm", "WeightKg", "DiagAge","LOfDiabDiag", "LabHbA1cDt", "Hba1c", "Age"]]

        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_RTC["Database"] = "RT-CGM"
        return df_RTC
    
    
    def df_T1GDUJA():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_T1G = smart_read("Data/datasets_for_T1D/T1GDUJA/glucose_data.csv")

        # converts timstamps to datetime depending on the format
        try: 
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"], format="%Y-%m-%d %H:%M:%S")
        # column names are renamed for semantic equality
        df_T1G = df_T1G.rename(columns={"sgv": "GlucoseCGM"})
        # adds the database name to the patient ID to enable reidentification 
        df_T1G["PtID"] = "T1GDUJA"
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_T1G = df_resample(df_T1G, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM")
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_T1G["Database"] = "T1GDUJA"
        return df_T1G
    

    def df_AZT1():

        # Path to the root folder
        root_folder = "Data/datasets_for_T1D/AZT1D 2025/CGM Records"  

        # List to hold each DataFrame
        all_dfs = []
        i = 1

        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith(".csv"):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        df = pd.read_csv(file_path)
                        df["PtID"] = i 
                        all_dfs.append(df)
                        i = i + 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Concatenate all DataFrames into one
        if all_dfs:
            df_AZT125 = pd.concat(all_dfs, ignore_index=True)
        else:
            print("No CSV files found.")
        df_AZT125["ts"] = pd.to_datetime(df_AZT125["EventDateTime"])
        df_AZT125 = df_AZT125.rename(columns={"CGM": "GlucoseCGM"})
        df_AZT125["PtID"] = df_AZT125["PtID"].astype(str) + "_AZT1"
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_AZT125 = df_resample(df_AZT125, timestamp = "ts", frequency= frequency, mode="glucose", fillid = "PtID", value = "GlucoseCGM")
        # adds a "Database" column with the name of the Dataset to enable reidentification
        # reads dataframe with demographic information
        df_AZT125_info = smart_read("Data/datasets_for_T1D/ayt1_demo.csv")
        df_AZT125_info["PtID"] = df_AZT125_info["PtID"].astype(str) + "_AZT1"
        df_AZT125 = pd.merge(df_AZT125, df_AZT125_info, on=["PtID"], how="left")

        # extracts the date of initial measurement
        df_AZT125["initial_date"] = df_AZT125.groupby("PtID")["ts"].transform("min")
        # calculates the birthyear
        df_AZT125["Birthyear"] = df_AZT125["initial_date"].dt.year - df_AZT125["Age"] 
        # drop age column 
        df_AZT125 = df_AZT125.drop("Age", axis=1)
        # computes the age of each year based on the birthyear
        df_AZT125["Age"] = df_AZT125["ts"].dt.year - df_AZT125["Birthyear"]

        df_AZT125["Database"] = "AZT1"
        return df_AZT125
    



    # this function calls the target functions reading the datasets which are given as a list and returns them as a list of dataframes 
    def try_call_functions(functions):
        # dictionary is initialized
        combined_df_dict = {}
        # each function is called seperately
        for func in functions:
            # try to read the data 
            try:
                dataframe = func()
                combined_df_dict[func.__name__] = dataframe
            # if an error occurs print error and continue to read the next function
            except Exception as e:
                print(f"Error in {func.__name__}(): {e}")
        # dataframes are combined and stored as a list 
        combined_df_list = list(combined_df_dict.values())
        return combined_df_list
    
    # if the integrated datasets of publicly available databases are found, only readsrestricted datasets
    if (read_all == False):
        datasets = [df_granada, df_diatrend]
    # if the integrated dataset is not found, reads all datasets separately
    elif(read_all == True):
        datasets = [df_granada, df_diatrend, df_city, df_dclp, df_pedap, df_replace, df_sence, df_shd, df_wisdm, df_shanghai, df_hupa,df_d1namo, df_DDATSHR, df_rtc, df_T1GDUJA]
    
    # calls all functions to return a list of all datasets 
    combined_df_list = try_call_functions(datasets)
    # returns a list of dataframes
    return combined_df_list



def combine_data(modus, restricted_list, columns_to_check = ["Age", "Sex"]):
    """
    This function integrates all dataframes into one database depending on the defined modus 
    Parameters:
    - modus: can be either 1, 2, or 3. 
        - 1 is for the main database and integrates all dataframes which include CGM values
        - 2 is for subdatabase I and integrates all dataframes which include CGM values and demographics of age and sex
        - 3 is for subdatabase II and integrates all dataframes which include CGM values and HR data
    - retricted_list is the list of datasets which were not shared due to licensing restrictions. These need to be loaded and integrated seperately
    - columns_to_check are default values to set age groups if modus two is applied
    Output: returns the integrated subset of restricted data 
    """
    for i, df in enumerate(restricted_list):
        restricted_list[i] = df.replace("NR", np.nan)

    # this function concatenates the dataframes 
    def concat_rows_on_columns(dfs, columns):
    
        # selects only the specified columns from each dataframe
        dfs = [df[columns] for df in dfs]
        # concatenates all dataframes vertically 
        result = pd.concat(dfs, ignore_index=True)
        # removes subjects who only includes nan values in the "GlucoseCGM" column
        subjects_to_keep = result.groupby("PtID")["GlucoseCGM"].transform(lambda x: not x.isna().all())
        # only includes subjects with columns
        result = result[subjects_to_keep]

        # if columns to check are all within the dataframe columns 
        if all(col in result.columns for col in columns_to_check):
            # removes nan values since the final datasets should have all columns included for each subject
            df_cleaned = result.dropna(subset=columns_to_check)
            # groups ages into age ranges 
            # returns the cleaned and preprocessed dataset
            return df_cleaned
        else:
            # returns the cleaned and preprocessed dataset
            return result
        
        
    # Validates modus
    allowed_values = [1, 2, 3, 4,5]
    if modus not in allowed_values:
        print("Invalid input. Please enter: 1, 2, 3 or 4")
        return

    # Sets columns based on modus
    if modus == 1:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Database"]
    elif modus == 2:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Age", "Sex", "Database"]
    elif modus == 2:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "HR", "Database"]
    elif modus == 4:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Age", "Sex", "Hba1c", "DiagAge", "Race", "HeightCm", "WeightKg", "Database"]
   
    # Filters and combines
    filtered_dfs = [df for df in restricted_list if all(col in df.columns for col in columns_to_keep)]

    combined_df = concat_rows_on_columns(filtered_dfs, columns=columns_to_keep)

    # Removes duplicate entries
    df_sorted = combined_df.sort_values(by=["PtID", "ts"])
    df_sorted = df_sorted.drop_duplicates(subset=["PtID", "ts"], keep="first")

    # Returns the combined_df
    return df_sorted