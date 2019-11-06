import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import FeatureAgglomeration

#Step 1
def import_csv_file(filepath):
    voters_data = pd.read_csv(filepath, low_memory=False)
    print("🗃️ Data imported!")
    return voters_data

    #Todo: clean up "mixed types" error message
    #Todo: make sure correct dataset

#Step 2
def choose_subset(voters_data, subset):
    print("🔴 Number of observations in starting dataset: {}".format(voters_data.shape[0]))
    if subset == "2016":
        voters_subset = voters_data[voters_data['weight_2016'].isna() == False]
        print("🔻 Selected subset of December 2016 survey respondents")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    elif subset == "2017":
        voters_subset = voters_data[voters_data['weight_2017'].isna() == False]
        print("🔻 Selected subset of July 2017 survey respondents")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    elif subset == "panel":
        voters_subset = voters_data[voters_data['weight_panel'].isna() == False]
        print("🔻 Selected subset of May 2018 survey respondents who were part of the original panel")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    elif subset == "overall":
        voters_subset = voters_data[voters_data['weight_overall'].isna() == False]
        print("🔻 Selected subset of *all* May 2018 survey respondents: original panelists, Latino oversample, and 18-24 year old oversample")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    elif subset == "latino":
        voters_subset = voters_data[voters_data['weight_latino'].isna() == False]
        print("🔻 Selected subset of May 2018 respondents who are part of the Latino oversample")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    elif subset == "18_24":
        voters_subset = voters_data[voters_data['weight_18_24'].isna() == False]
        print("🔻 Selected subset of May 2018 respondents who are part of the 18-24 year old oversample")
        print("⭕ Number of observations in selected subset: {}".format(voters_subset.shape[0]))
    else:
        voters_subset = "⚠️ No subset selected. Set the parameter `subset` to be one of the following strings: '2016', '2017', 'panel', 'overall', 'latino', '18_24'"
        print(voters_subset)
    return voters_subset

    #Todo: simplify

#Step 4
def apply_weights(voters_data, subset, magnitude=3):

    subset_weights = "weight_{}".format(subset)

    if subset_weights not in voters_data.columns:
        print("⚠️ Set the parameter `subset` to be one of the following strings: '2016', '2017', 'panel', 'overall', 'latino', '18_24'")

    weights = voters_data[subset_weights]
    new_index = []
    new_units = []
    for index_no in tqdm(weights.index, desc="Expanding data according to weight"):
        units = int(round(weights[index_no] * magnitude))
        for unit in range(0, units):
            new_unit = dict(voters_data.loc[index_no])
            new_index_no = round(index_no + unit*.001, ndigits=3)
            new_index.append(new_index_no)
            new_units.append(new_unit)
    reconstituted_voters_data = pd.DataFrame(new_units, index=new_index, columns=voters_data.columns)
    print("🔼🔼🔺🔼🔺🔺 Data unpacked according to provided weights.")
    return reconstituted_voters_data

###############################
# Choose Features of Interest #
###############################

class FeaturesPicker():
    def __init__(self, IssueImportance = False,
                 FeelingThermometer = False,
                 Favorability = False,
                 InstitutionalConfidence = False):

        self.feature_picker_list = [{'category' : 'Issue Importance',
                                    'import' : IssueImportance,
                                    'prefixes' : ('imiss_'),
                                    'keywords' : [],
                                    'exceptions' : []},
                               {'category' : 'Feeling Thermometer',
                                   'import' : FeelingThermometer,
                                   'prefixes' : ('ft'),
                                   'keywords' : [],
                                   'exceptions' : []},
                               {'category' : 'Favorability',
                                   'import' : Favorability,
                                   'prefixes' : ('fav'),
                                   'keywords' : [],
                                   'exceptions' : []},
                               {'category' : 'Institutional Confidence',
                                   'import' : InstitutionalConfidence,
                                   'prefixes' : ('inst_'),
                                   'keywords' : [],
                                   'exceptions' : []}
                              ]


    def clean(self, voters_data, RandomizationFeatures = True,
                 StartEndTimes = True):

        feature_cleaner_list = [{'category' : 'Randomization Features',
                                    'import' : RandomizationFeatures,
                                    'prefixes' : (),
                                    'keywords' : ['rnd'],
                                    'exceptions' : []},
                               {'category' : 'Start/End Times',
                                   'import' : StartEndTimes,
                                   'prefixes' : ('starttime_', 'endtime_'),
                                   'keywords' : [],
                                   'exceptions' : []},
                              ]

        for feature_cleaner in feature_cleaner_list:
            if feature_cleaner['import'] == True:
                for feature in list(voters_data.columns):
                    if feature.startswith((feature_cleaner['prefixes'])) == True:
                        voters_data.pop(feature)
                    for keyword in feature_cleaner['keywords']:
                        if keyword in feature:
                            voters_data.pop(feature)

        return voters_data

    def process(self, voters_data):
        features_of_interest = []
        for feature_picker in self.feature_picker_list:
            if feature_picker['import'] == True:
                for feature in list(voters_data.columns):
                    if feature.startswith((feature_picker['prefixes'])) == True:
                        features_of_interest.append(feature)
        cleaned_features_of_interest = self.clean(voters_data[features_of_interest])
        selected_features_readable = ", ".join([x['category'] for x in self.feature_picker_list if x['import'] == True])
        print("🔵 {} features out of {} selected, relevant to: {}".format(len(list(cleaned_features_of_interest.columns)),len(list(voters_data.columns)),selected_features_readable))
        return cleaned_features_of_interest

########################
# NARROW DOWN FEATURES #
########################

important_features = ["case_identifier", "caseid", "weight_panel", "weight_latino", "weight_18_24", "weight_overall", "cassfullcd", "starttime", "endtime"]

def check_feature_exists(reconstructed_feature, feature_list):
    """Subfunction for condense_by_most_recent_feature(), perhaps turn into lambda"""
    if reconstructed_feature in feature_list:
        match = True
    else:
        match = False
    return match

def reconstruct_feature(x):
    """Subfunction for condense_by_most_recent_feature(), perhaps turn into lambda"""
    return x.index + x.name # Source: https://stackoverflow.com/a/43654808

def condense_by_most_recent_feature(voters_data):
    condensed_data = voters_data.copy()
    wave_suffix_list = ["_baseline", "_2012", "_2016", "_2017", "_2018"]
    feature_list = list(voters_data.columns)
    feature_root_set = set()
    for wave_suffix in wave_suffix_list:
        for feature in feature_list:
            if feature.endswith(wave_suffix):
                feature_root = feature.replace(wave_suffix, "")
                feature_root_set.add(feature_root)
    feature_grid = pd.DataFrame(index=feature_root_set, columns=wave_suffix_list)

    feature_grid = feature_grid.apply(reconstruct_feature)
    feature_grid = feature_grid.applymap(lambda x : check_feature_exists(x, feature_list))
    feature_grid['latest'] = ""

    for feature in feature_root_set:
        for wave in wave_suffix_list:
            if feature_grid.loc[feature,wave] == True:
                feature_grid.loc[feature,"latest"] = wave

    most_recent_features = []
    for feature in feature_root_set:
        most_recent_feature = feature + feature_grid.loc[feature,"latest"]
        most_recent_features.append(most_recent_feature)

    features_to_preserve = most_recent_features + important_features

    features_to_drop = []
    for feature in feature_list:
        if feature not in features_to_preserve:
            features_to_drop.append(feature)
    features_to_drop

    for feature_to_drop in features_to_drop:
        condensed_data.pop(feature_to_drop)

    print("📉 Number of features reduced from {} to {}.".format(len(feature_list), len(list(condensed_data.columns))))

    return condensed_data

#########################
# Manage missing values #
#########################

def remove_low_response_respondents(voters_data, threshold_factor = 10):
    beginning_number_respondents = (len(voters_data))
    voters_data['na'] = voters_data.apply(lambda x: sum(x.isna()), axis=1)
    threshold = (len(voters_data.columns) // threshold_factor) + 1 #arbitrary threshold for missing values, rounded up
    voters_data = voters_data[voters_data['na'] <= threshold]
    voters_data = voters_data.drop(columns='na')
    number_respondents_removed = beginning_number_respondents - len(voters_data)
    print("🔲 {} respondents removed for having {} or more missing values.".format(number_respondents_removed, threshold))
    return voters_data

def impute_nas(voter_data):
    samesies = 0
    not_samesies = 0
    for column_name in tqdm(voter_data.columns, desc="Imputing"):
        feature_mode = float(voter_data.mode()[column_name])
        for index_value in voter_data.index:
            if np.isnan(voter_data[column_name][index_value]) == True:
                respondent_mode = float(sum(voter_data.loc[index_value].mode()))/len(voter_data.loc[index_value].mode())
                if respondent_mode == feature_mode:
                    samesies += 1
                    voter_data.loc[index_value,column_name] = respondent_mode
                elif respondent_mode != feature_mode:
                    not_samesies += 1
                    voter_data.loc[index_value,column_name] = float(round((feature_mode + respondent_mode)/2))
    print("🔳 {} values imputed by modal value of feature (= modal value of respondent),\n and {} values imputed by mean value of modal value of feature and (mean) modal value(s) of respondent.".format(samesies, not_samesies))
    return voter_data

#########################
# Feature Agglomeration #
#########################

def feature_agglomeration(voters_data, n, rounding=False):
    featagg = FeatureAgglomeration(n_clusters=n)
    featagg.fit(voters_data)
    condensed = featagg.transform(voters_data)

    feature_groups_map = dict(zip(voters_data.columns, featagg.labels_))
    feature_groups_nos = []
    for feature_group_key in feature_groups_map:
        feature_groups_nos.append(feature_groups_map[feature_group_key])
    feature_groups_nos

    group_labels = []
    for feature_group_no in set(feature_groups_nos):
        group_label = ""
        for feature_groups_key in feature_groups_map:
            if feature_groups_map[feature_groups_key] == feature_group_no:
                group_label = group_label + feature_groups_key + ", "
        group_labels.append(group_label[0:-2])
    group_labels

    voters_agglomerated = pd.DataFrame(condensed, columns=group_labels, index=voters_data.index)
    if rounding == True:
        voters_agglomerated = voters_agglomerated.applymap(lambda x : round(x))
    print("🔹→💠←🔹 {} features agglomerated into {} hybrid features.".format(len(voters_data.columns),len(voters_agglomerated.columns)))
    return voters_agglomerated

#########################
# Respondent Bias       #
#########################

def importance_scale_inverter(x):
    if x == 4.0:
        return 0.0
    elif x == 3.0:
        return 1.0
    elif x == 2.0:
        return 2.0
    elif x == 1.0:
        return 3.0

def importance_scale_proportionizer(voters_data):
    voters_data['total'] = voters_data.apply(lambda x: x.sum(), axis=1)
    proportion_adjusted = voters_data.apply(lambda x: x / x['total'] if x['total'] != 0 else 1 / (len(voters_data.columns) - 1), axis=1)
        # else code solves problem created by importance_scale_inverter() for voters who respond "0"/"Unimportant" for every issue
    voters_data.drop('total', axis=1, inplace=True)
    proportion_adjusted.drop('total', axis=1, inplace=True)
    print("⏫ Proportional scale calculated.")
    return proportion_adjusted

def investigate_variable_by_subset(guide_dict, variable_of_interest, subset_of_interest):
    #print(guide_dict[variable_of_interest]['question'])
    key_labels = {guide_dict[variable_of_interest]['responses'][i]['numeric']:guide_dict[variable_of_interest]['responses'][i]['label'] for i in range(len(guide_dict[variable_of_interest]['responses']))}
    statistic = round(subset_of_interest[variable_of_interest].value_counts()/len(subset_of_interest[variable_of_interest]), 2)
    statistic.index = statistic.index.map(key_labels)
    return statistic

def investigate_variable_across_clusters(guide_dict, variable_of_interest, full_data_clusters):
    print(guide_dict[variable_of_interest]['question'])
    cluster_no = 0
    for cluster in full_data_clusters:
        cluster_no += 1
        print("\nCluster #{} breakdown:".format(cluster_no))
        statistic = investigate_variable_by_subset(guide_dict, variable_of_interest, cluster)
        print(statistic)
    return
