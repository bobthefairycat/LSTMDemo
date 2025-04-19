# Note
The data required for this code to run is not provided due to it being of a sensitive nature.
* `main.py` is the main utility for loading in the data, conducting the train-test split, training models (with cross-validation), and plotting train/test loss curves.
* `model.py` constructs the LSTM model with the appropriate PyTorch class.
* `model_training.py` provides utilities to segment the data for cross-validation and a function to run the desired number of training epochs; native PyTorch data train-test splitting utilities did not support some of the flexibility I wanted.

# Requirements

+ Python 3.8 or above
+ All packages in `requirements.txt`
+ Do **not** upload any of the data to this git repository

# Program Flow

1. Run the file `data_loader.py` to load in the data. **You only need to do this once per set of data loaded in**.
   * EMA data in the files `data/raw/*.csv` are loaded in with corresponding PHQ-9 scores from `data/raw/depression_scores.csv` and EMA start dates from `data/raw/ema_start_dates.csv`
     * You can change these directories through the global variables in the file.
   * Loaded EMA data is stored in a Python dictionary; this dictionary then _pickled_ (saved to a Python binary format). The pickle file is saved as `data/pickled/2week_ema_data.pickle`.
      * Change the pickle file through the global variable `PICKLE_DATA_FNAME`
   
      Resulting directory structure:
      ```
      data
      |____ pickled
      |     |____ 2week_ema_data.pickle
      |
      |____ raw
            |____ ema_csv
            |     |____ ...
            |
            |____ depression_scores.csv
            |____ ema_start_dates.csv
            
      ```
  
      To load in all participant data in the future, you can simply load in the pickled file given by `<PICKLE_DATA_FNAME>` (it should be much faster than reading in the raw files again). 
     
      The resulting data dumped to the pickle file has the form:
      
      ```
      {
         <participant_id>:  {
             "ema": pd.DataFrame of EMA data,
             "depression_scores": pd.DataFrame of PHQ-9 scores,
             "start_date": datetime object storing start date of 2-week EMA protocol,
             "two_weeks_date": datetime object storing two weeks after start_date 
         }
      }
      ```
     If you want to look at the list of participant ids, you can simply examine the keys of the resulting dictionary (via a call to the `keys()` method.)
2. See `basic_example.py` for the steps to extract secondary features from the data (using modules `features/feature_metrics.py` and `features/feature_extraction.py`).

Terminology summary:

+ Measure: something _measured_ from the participant; e.g., an entry in the EMA survey spreadsheet. Also referred to as a "mood measure" or "raw mood measure", with "raw" indicating that it is directly measured and not processed in any way. For example, the survey response with column label "mood1" is a measure. 
+ Metric: a calculation or computation done on a set of values to obtain a new (set) of numerical value(s). The metric is the operation itself, e.g., "dividing all values by 2" or "taking the average across time".
+ Secondary feature: the resulting numerical value obtained from applying a metric to raw measures (or other secondary features).
+ Feature: Umbrella term combining measures and secondary features. The set of all features is the union of the raw measures and the secondary features derived through applying metrics.

If you find any inconsistencies with the terminology usage within the code, please let me know ASAP.

Feature extraction summary:

+ `feature_extraction.py` does the heavy lifting on extracting secondary features as per the docstring of `extract_single_participant_features`. It returns an instance of `AppliedMetrics` from `feature_metrics.py`.
    + `AppliedMetrics` from `feature_metrics.py` stores information about the _metrics_ applied on all _mood measures_ for a single participant.

This readme to be updated with further information on how the data is formatted for model training.



