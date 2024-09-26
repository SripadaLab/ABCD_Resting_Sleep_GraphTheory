import pandas as pd
import numpy as np
from scipy import stats

def gather_sleep_data_for_EFA(path_to_ABCD_release_4, output_file='sleep_factor_year2_weekdays.csv'):
    """
    Gathers sleep data from multiple sources (Parent Sleep Disturbance, Munich chronotype, and Fitbit data)
    for 2-year follow-up in the ABCD Study and merges them into a single dataframe. 
    The final dataframe is saved as a CSV file.
    
    This saved CSV file should be loaded into R for the EFA as follows:
    ################
      EFA (R Code)
    ################
    df = read.csv('./sleep_factor_year2_weekdays.csv')
    df = df[c('fit_ss_sleepperiod_minutes', 'mctq_sdw_calc', 'sleepdisturb1_p')]
    df <- na.omit(df)
    library(psych)
    fa_fit <- fa(df, nfactors=1, rotate="none")
    fa_fit
    ################

    Parameters:
    -----------
    path_to_ABCD_release_4 : str
        The base path to the ABCD study release 4 dataset.
    
    output_file : str, optional
        The output file path where the processed dataframe will be saved as a CSV file.
        Default is 'sleep_factor_year2_weekdays.csv'.

    Returns:
    --------
    efa_df : pandas.DataFrame
        The merged dataframe containing processed sleep data from all sources.
    """

    merge_cols = ['subjectkey']

    # Load and process parent sleep disturbance data
    parent_sleep_df = pd.read_csv(f"{path_to_ABCD_release_4}/abcd_sds01.txt", delim_whitespace=True, skiprows=[1])
    parent_sleep_df = parent_sleep_df[parent_sleep_df.eventname == '2_year_follow_up_y_arm_1']
    parent_sleep_df['sleepdisturb1_p'] = parent_sleep_df.sleepdisturb1_p.astype(np.float32)
    parent_sleep_mapping = {
        1: 10, 
        2: 8.5, 
        3: 7.5, 
        4: 6, 
        5: 5
    }
    parent_sleep_df = parent_sleep_df.replace({"sleepdisturb1_p": parent_sleep_mapping})
    parent_sleep_df = parent_sleep_df[merge_cols + ['sleepdisturb1_p']]

    # Load and process Munich chronotype data
    munich_df = pd.read_csv(f"{path_to_ABCD_release_4}/abcd_mcqc01.txt", delim_whitespace=True, skiprows=[1])
    munich_df = munich_df[munich_df['eventname'] == '2_year_follow_up_y_arm_1']
    munich_df = munich_df[merge_cols + ['mctq_sdw_calc']].dropna()
    munich_df['mctq_sdw_calc'] = munich_df['mctq_sdw_calc'].astype(np.float32)
    munich_df['mctq_sdw_calc'] = stats.zscore(munich_df.mctq_sdw_calc.values.flatten())
    munich_df = munich_df[munich_df.mctq_sdw_calc.abs() < 5]
    munich_df = munich_df[merge_cols + ['mctq_sdw_calc']]

    # Load and process Fitbit sleep data
    sleep_fitbit_df = pd.read_csv(f"{path_to_ABCD_release_4}/abcd_fbdss01.txt", delim_whitespace=True, skiprows=[1])
    sleep_fitbit_df = sleep_fitbit_df[sleep_fitbit_df.eventname == '2_year_follow_up_y_arm_1']
    sleep_fitbit_df = sleep_fitbit_df[sleep_fitbit_df.fit_ss_weekend_ind.astype(int) == 0]
    
    counts = sleep_fitbit_df[['subjectkey'] + ['fit_ss_sleepperiod_minutes']].value_counts('subjectkey')
    good_subs = counts[counts >= 3].index
    sub_vals = []
    
    for sub in good_subs:
        sub_df = sleep_fitbit_df[sleep_fitbit_df.subjectkey == sub]
        assert len(sub_df.eventname.unique()) == 1
        vals = [sub, sub_df.eventname.unique()[0]]
        for col in ['fit_ss_sleepperiod_minutes']:
            arr = sub_df[[col]].values.astype(np.float32).flatten()
            arr[arr == 0] = np.nan
            vals.append(np.nanmean(arr))
        sub_vals.append(vals)

    fitbit_sleep_df = pd.DataFrame(sub_vals)
    fitbit_sleep_df.columns = ['subjectkey', 'eventname'] + ['fit_ss_sleepperiod_minutes']
    fitbit_sleep_df = fitbit_sleep_df.dropna()
    fitbit_sleep_df = fitbit_sleep_df[fitbit_sleep_df.eventname == '2_year_follow_up_y_arm_1']
    fitbit_sleep_df = fitbit_sleep_df[merge_cols + ['fit_ss_sleepperiod_minutes']].dropna()
    fitbit_sleep_df['fit_ss_sleepperiod_minutes'] = stats.zscore(fitbit_sleep_df.fit_ss_sleepperiod_minutes.values.flatten())
    fitbit_sleep_df = fitbit_sleep_df[fitbit_sleep_df.fit_ss_sleepperiod_minutes.abs() < 5]
    fitbit_sleep_df = fitbit_sleep_df[merge_cols + ['fit_ss_sleepperiod_minutes']]

    # Merge all dataframes
    efa_df = fitbit_sleep_df.merge(munich_df, on=merge_cols, how='outer').merge(parent_sleep_df, on=merge_cols, how='outer')

    # Save to CSV
    efa_df.to_csv(output_file, index=False)

    return efa_df
