"""
Module with model class for comparing string describing insurance plan to plans in Ribbon Health
taxonomy.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import regex
from sklearn.linear_model import LogisticRegression
from . import simvec2score

DATA_DIR = '../Data'
DATA_FILE_NAME = 'Insurance Mappings RIbbon.xlsx'
TAXONOMY_SHEETNAME = 'Ribbon Insurance Taxonomy'
MAPPED_PLANS_SHEETNAME = 'Confirmed Mappings to UUIDs'
STATES_ABBREV_FILE_NAME = 'list_states_US.txt'

DEFAULT_FEATURE_NAMES = ['carrier_association',
'carrier_brand', 'carrier_name', 'state', 'plan', 'type', 'metal']

class PlanComparisonModel():
    """Model that compares input text string to insurance plans and finds best matches.

    Important attributes:
        taxonomy (df): taxonomy of insurance plans.
        mapped_plans (df): input strings with confirmed mappings to plans.
        vocab (dict): dictionary of sets of words making up the vocabulary of each plan property.
        aliases_dict (dict): look-up table that maps terms to standardized form.
        state_aliases_dict (dict): look-up table that maps states to standardized form.
    """
    def __init__(self, features=DEFAULT_FEATURE_NAMES,data_dir=DATA_DIR, \
                       file_name=DATA_FILE_NAME,tax_sheet_name=TAXONOMY_SHEETNAME, \
                       mapped_plans_sheetname=MAPPED_PLANS_SHEETNAME, \
                       states_file_name=STATES_ABBREV_FILE_NAME):
        """Load taxonomy, prepare data vocabulary, load vec2score model, etc."""

        self.data_dir = data_dir
        self.file_name = file_name
        self.tax_sheet_name = tax_sheet_name
        self.mapped_plans_sheetname = mapped_plans_sheetname
        self.states_file_name = states_file_name
        self.features = features
        self.feature_match_names = [feature + '_match' for feature in self.features]
        self.feature_match_names += ['input_string_match']
        self.mapped_plans_loaded = False
        self.build_aliases_dictionary()
        self.load_taxonomy()
        self.build_vocabulary()
        self.sim_vec_to_score = simvec2score.SimilarityVectorToScore()
        #load the model mapping similarity vector to score from file:
        self.sim_vec_to_score.load_model()

    def build_aliases_dictionary(self):
        """Construct mapping from alternative spellings to standardized form."""

        state_aliases_dict = {}
        #load state names:
        fname = os.path.join(self.data_dir, self.states_file_name)
        states_abbrev = pd.read_table(fname, skiprows=1, header=None)
        states_abbrev.columns = ['state_name', 'state_abbrev']
        for (i, row) in states_abbrev.iterrows():
            state_aliases_dict[row['state_name'].lower()] = row['state_abbrev'].lower()

        #other aliases by hand:
        aliases_dict = {}
        aliases_dict['bcbs'] = 'blue cross blue shield'
        aliases_dict['compl'] = 'complete'
        aliases_dict['ibc'] = 'independence blue cross'
        aliases_dict['unitedhealthcare'] = 'united healthcare'
        aliases_dict['health care'] = 'healthcare'
        aliases_dict['comp'] = 'compensation'
        aliases_dict['\+'] = 'plus'
        aliases_dict['ntwk'] = 'network'
        aliases_dict['healthamerica'] = 'health america'
        aliases_dict['advantageplus'] = 'advantage plus'
        aliases_dict['compbenefits'] = 'compensation benefits'
        aliases_dict['directaccess'] = 'direct access'
        aliases_dict['workers compensation insurance'] = 'workers compensation'

        self.aliases_dict = aliases_dict
        self.state_aliases_dict = state_aliases_dict

    def load_taxonomy(self):
        """Load insurance taxonomy into dataframe."""

        fname = os.path.join(self.data_dir, self.file_name)
        taxonomy_orig = pd.read_excel(fname, self.tax_sheet_name)
        self.taxonomy_orig = taxonomy_orig
        taxonomy = taxonomy_orig.copy()
        taxonomy.drop(['display_name', 'network'], axis=1, inplace=True)
        taxonomy.rename(columns={'Ribbon Insurance UUID': 'UUID', \
        'plan_name':'plan', 'plan_type':'type', 'metal_level':'metal'}, inplace=True)

        for col in taxonomy.columns:
            if col != 'UUID':
                taxonomy[col] = taxonomy[col].str.lower()

        for col in ['carrier_association','carrier_brand','carrier_name','plan']:
            for state, abbrev in self.state_aliases_dict.items():

                contains_ofstate = taxonomy[col].str.contains(r'of '+state, na=False)
                taxonomy[col] = taxonomy[col].str.replace(r'of '+state, '')
                contains_state = taxonomy[col].str.contains(state, na=False)
                taxonomy[col] = taxonomy[col].str.replace(state, '')

                contains_either = (contains_ofstate) | (contains_state)
                taxonomy.loc[(contains_either) & (taxonomy.state.isnull()), 'state'] = abbrev

            state, abbrev = 'ny', 'ny'
            contains_ofstate = taxonomy[col].str.contains(r'of '+state, na=False)
            taxonomy[col] = taxonomy[col].str.replace(r'of '+state, '')
            taxonomy.loc[(contains_ofstate) & (taxonomy.state.isnull()), 'state'] = abbrev
            for alias in self.aliases_dict:
                taxonomy[col] = taxonomy[col].str.replace(r'\b'+alias+r'\b', self.aliases_dict[alias])
            taxonomy[col] = taxonomy[col].str.strip()

        #if plan name does not add info, treat it as NaN:
        taxonomy.plan.replace('healthcare', np.nan, inplace=True)
        taxonomy.index = np.arange(len(taxonomy))
        self.taxonomy = taxonomy

    def build_vocabulary(self):
        """Construct vocabulary for each feature (e.g. 'carrier_association') based on taxonomy."""

        vocabs = {}
        for feature in ['carrier_association','carrier_brand','carrier_name', \
            'state', 'plan', 'type']:
            feature_strings = list(self.taxonomy.loc[self.taxonomy[feature].notnull(), feature])
            feature_vocab = set(feature_strings)
            vocabs[feature] = feature_vocab
        vocabs['metal'] = set(['gold', 'silver', 'bronze', 'platinum'])
        self.vocab = vocabs

    def load_mapped_plans(self):
        """Load already mapped inputs into dataframe."""

        fname = os.path.join(self.data_dir, self.file_name)
        mapped_plans = pd.read_excel(fname, self.mapped_plans_sheetname)
        mapped_plans.drop(['Carrier', 'Plan'], axis=1, inplace=True)
        mapped_plans.rename(columns={'Ribbon Insurance UUID': 'UUID_known', \
        'Carrier - Plan':'input_string'}, inplace=True)
        mapped_plans.index = np.arange(len(mapped_plans))

        mapped_plans = pd.merge(mapped_plans, self.taxonomy, how='left',left_on='UUID_known',right_on='UUID')
        mapped_plans.drop('UUID_known', axis=1, inplace=True)
        self.mapped_plans = mapped_plans
        self.mapped_plans_loaded = True

    @staticmethod
    def _find_matches_in_vocab(input_string, feature_vocab):
        """Find list of all terms in vocabulary of individual feature that are
        a substring of input_string.
        """

        matches = []
        for entity_string in feature_vocab:
            match = regex.search(r'\b(' + entity_string + r'){e<=0}\b', input_string)
            if match:
                matches.append(entity_string)
        return matches

    def clean_input_string(self, input_string):
        input_string = input_string.lower()
        input_string = re.sub(r'\bof\b', '', input_string)
        input_string = re.sub(r"[-\)\(\/\s\']+", ' ', input_string)
        #input_string.replace('[-\)\(\s]+', ' ')
        for alias in self.aliases_dict:
            input_string = re.sub(r'\b'+alias+r'\b', self.aliases_dict[alias], input_string)
        for alias in self.state_aliases_dict:
            input_string = re.sub(alias, r' '+self.state_aliases_dict[alias]+r' ', input_string)
        input_string = re.sub(r'\s+', ' ', input_string)
        input_string = input_string.strip()

        return input_string

    def input_string_to_feature_dict(self, input_string):
        """Convert input string to dictionary of features.

        Each key represents a feature ('carrier_association', etc), and each value is the list
        of strings from the vocabulary that are a substring of input_string.
        """

        input_string_clean = self.clean_input_string(input_string)
        input_dict = {}
        for feature in self.features:
            input_dict[feature] = self._find_matches_in_vocab(input_string_clean,\
                                                        self.vocab[feature])
        input_dict['input_string_clean'] = input_string_clean
        return input_dict

    def similarity_vector(self, input_dict, plan_dict):
        """Compute similarity vector.

        For each feature, it tells us if the input_dict matches the plan_dict or not.
        The final feature counts what fraction of distinct words in the input string
        is accounted for in the plan description.

        Arguments:
            input_dict (dict): Dictionary characterizing input_string. each key is a feature,
                each value a list of substrings of input_strings that are part of the vocabulary
                of that feature.
            plan_dict (dict): Dictionary characterizing a plan in the taxonomy. Contains a string
                for each feature.

        Returns:
            sim_vec (np.array): array with 1's and 0's indicating whether each feature matches
                the corresponding feature in  the plan from the taxonomy
        """

        sim_vec = np.zeros((len(self.features)) + 1)
        matching_words = []
        for i, feature in enumerate(['carrier_association','carrier_brand','carrier_name', \
            'state', 'plan', 'type']):
            plan_feature = plan_dict[feature]
            if isinstance(plan_feature, str):
                if plan_feature in input_dict[feature]:
                    sim_vec[i] = 1
                matching_words += re.split(r'[\s\,]+', plan_feature.strip())
            #if type(plan_dict[feature]) == type(np.nan):
            elif np.isnan(plan_feature) & (not input_dict[feature]):
                    sim_vec[i] = 0.5
        #do 'metal' separately:
        metal = plan_dict['metal']
        if isinstance(metal, str): # type(metal) == '' not np.isnan(metal):
            plan_set = set(re.split(r'[\s\,]+', metal.strip()))
            input_set = set(input_dict['metal'])
            sim_vec[self.features.index('metal')] = int(input_set <= plan_set) # hardcoded index for now (improve that)
            matching_words += list(input_set) # re.split(r'[\s\,]+', list(input_set).strip())
            #print(input_set, plan_set, int(input_set < plan_set))
            #jaccard_similarity(plan_set, input_set)
        matching_word_set = set(matching_words)
        input_word_set = set(re.split(r'[\s\,]+', input_dict['input_string_clean']))
        #print(matching_word_set)
        #print(input_word_set)
        sim_vec[len(self.features)] = len(matching_word_set & input_word_set)/len(input_word_set)
        #print(sim_vec[len(self.features)])
        #category for overall match:
        #plan_words = [re.split(r'[\s,]+', plan_dict[feature]) for feature in self.input_features]

        return sim_vec

    @staticmethod
    def _taxonomy_row_to_plan_feature_dict(tax_row):
        """Convert row from taxonomy dataframe to dict defining a plan."""
        plan_dict = tax_row.to_dict()
        return plan_dict

    def UUID_to_plan_feature_dict(self, UUID):
        """Find and return plan_dict based on plan UUID."""
        plan_dict = self.taxonomy[self.taxonomy['UUID'] == UUID].iloc[0].to_dict()
        return plan_dict

    def index_to_plan_feature_dict(self, ix):
        """Find and return plan_dict based on index in DataFrame."""
        plan_dict = self.taxonomy.iloc[ix,:].to_dict()
        return plan_dict

    def compute_similarity_vectors(self, input_dict, taxonomy_or_subset):
        """Compute similarity vectors with plans in input taxonomy.

        Arguments:
            input_dict (dict): dictionary of features of input string.
            taxonomy_or_subset (df): taxonomy of plans.
        Returns:
            taxonomy_similarity (df): input taxonomy with similarity vecs.
        """

        taxonomy_similarity = taxonomy_or_subset.copy()
        #create new columns:
        for feature in self.feature_match_names:
            taxonomy_similarity[feature] = 0
        for (i, row) in taxonomy_or_subset.iterrows():
            plan_dict = self._taxonomy_row_to_plan_feature_dict(row)
            sim_vec = self.similarity_vector(input_dict, plan_dict)
            for n, feature in enumerate(self.feature_match_names):
                taxonomy_similarity.loc[i, feature] = sim_vec[n]

        return taxonomy_similarity

    def compute_similarity_scores(self, taxonomy_similarity):
        """Compute similarity scores from dataframe of similarity vectors."""

        taxonomy_similarity_scores = taxonomy_similarity.copy()
        taxonomy_similarity_sub = taxonomy_similarity[self.feature_match_names]
        taxonomy_similarity_scores['similarity_score'] = \
        taxonomy_similarity_sub.apply(lambda x: self.sim_vec_to_score.compute_score(x.values), axis=1)

        return taxonomy_similarity_scores

    def compute_comparison_features_match_nomatch(self, mapped_plans, nomatches_per_input):
        """Compute similarity vector dataframes for sets of matching plans and non-matching plansself.

        Arguments:
            mapped_plans (df): set of input strings with known matching plans.
            nomatches_per_input (int): number of non-matching plans per input to be included.
        Returns:
            X_train_match (np.array): each row is similarity vector with the matching plan.
            X_train_nomatch (np.array): each row is sim vec of randomly selected non-matching plan.
        """

        similarity_match = pd.DataFrame(columns=self.taxonomy.columns.tolist() \
            + self.feature_match_names)
        similarity_nomatch = pd.DataFrame(columns=self.taxonomy.columns.tolist() \
            + self.feature_match_names)
        for (ix, row) in mapped_plans.iterrows():
            input_string = row['input_string']
            UUID_known = row['UUID']
            input_dict = self.input_string_to_feature_dict(input_string)
            #first find the exact match:
            tax_sub_match = self.taxonomy[self.taxonomy['UUID'] == UUID_known]
            similarity_sub_match = self.compute_similarity_vectors(input_dict, tax_sub_match)
            similarity_match = pd.concat([similarity_match, similarity_sub_match])

            #then find the non-matching ones:
            tax_sub_nomatch = self.taxonomy[self.taxonomy['UUID'] != UUID_known]
            random_subset = np.random.choice(tax_sub_nomatch.index, \
                size=nomatches_per_input, replace=False)
            tax_sub_nomatch = tax_sub_nomatch.loc[random_subset,:]
            similarity_sub_nomatch = self.compute_similarity_vectors(input_dict, tax_sub_nomatch)
            similarity_nomatch = pd.concat([similarity_nomatch, similarity_sub_nomatch])

        X_train_match = similarity_match[self.feature_match_names].values
        X_train_nomatch = similarity_nomatch[self.feature_match_names].values
        return (X_train_match, X_train_nomatch)

    def index_to_mapped_input_plan(self, ix, mapped_plans):
        """Return plan information for plan in mapped_strings catalog at index ix."""

        input_string = mapped_plans.loc[ix, 'input_string']
        features = self.features + ['UUID'] # 'input_string'
        true_plan = mapped_plans.loc[ix, features].to_dict()
        input_feature_dict = self.input_string_to_feature_dict(input_string)
        plan_feature_dict = {}
        plan_feature_dict['input_string'] = input_string
        plan_feature_dict['true_plan'] = true_plan
        plan_feature_dict['input_feature_dict'] = input_feature_dict
        return plan_feature_dict

def jaccard_similarity(tokens1, tokens2):
    token_intersection = set(tokens1) & set(tokens2)
    token_union = set(tokens1) | set(tokens2)
    return len(token_intersection)/len(token_union)
