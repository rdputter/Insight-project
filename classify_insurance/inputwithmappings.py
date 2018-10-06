"""
Module defining class containing input string and its comparisons with taxonomy plans.
"""

import re
from . import planmapper

class InputWithMappings():
    """Class containing input text string, keeping track of comparisons with insurance plans
    and recommendations for best matching plan(s).

    Important attributes:
        input_string (str): string describing insurance plan.
        feature_dict (dict): dictionary of plan properties (matches in vocab).
        similarity (df): similarity vectors with plans in taxonomy.
        recommendations(df): as similarity, but ranked by similarity scores.
        num_rec (int): number of recommended matches.
        max_score (float): best similarity score in taxonomy.
    """
    def __init__(self, input_string = ''):
        self.input_string = input_string
        self.good_matches_found = False

    def obtain_features(self, mapper):
        self.feature_dict = mapper.input_string_to_feature_dict(self.input_string)

    def compute_similarity_vectors(self, mapper):
        self.obtain_features(mapper)
        self.similarity = mapper.compute_similarity_vectors(self.feature_dict, mapper.taxonomy)

    def compute_similarity_vectors_scores(self, mapper):
        self.obtain_features(mapper)
        self.similarity = mapper.compute_similarity_vectors(self.feature_dict, mapper.taxonomy)
        self.similarity = mapper.compute_similarity_scores(self.similarity)

    def get_recommendations(self, margin = 0.2, verbose=False):
        """Rank plans by similarity scores and output and compute summary values.

        Arguments:
            margin (float): plans with score > max score - margin are recommended matches.
            verbose (bool): if true, information about the matching plans is printed.
        """

        scores = self.similarity['similarity_score'].values
        recommendations = self.similarity.copy()
        recommendations.sort_values(by='similarity_score', ascending=False, inplace=True)
        max_score = scores.max()
        threshold = max_score - margin
        num_rec = (scores >= threshold).sum()

        self.recommendations = recommendations
        self.num_rec = num_rec
        self.max_score = max_score

        output_features = ['UUID','carrier_association','carrier_brand','carrier_name',
        'state', 'plan','type', 'metal','similarity_score']
        self.recommended_plan_dicts = self.recommendations[output_features].iloc[:num_rec,:].to_dict(orient='records')

        if max_score < 1.5: # assumes logreg
            if verbose:
                print('\nLOW SIMILARITY SCORES: CONSIDER ADDING A NEW ENTRY')
            self.good_matches_found = False
        else:
            if verbose:
                if num_rec == 1:
                    print('\nSINGLE RECOMMENDED MATCH FOUND!')
                else:
                    print('\nMULTIPLE RECOMMENDED MATCHES FOUND!')
            self.good_matches_found = True
        if verbose:
            print('top score = {}'.format(max_score))

    def recommend_taxonomy_entry(self):
        """Provide info on adding new entry to taxonomy if best similarity score low."""

        print('\n\nYOU MIGHT WANT TO ADD A NEW ENTRY TO THE TAXONOMY. HERE IS SOME INFORMATION TO CONSIDER.')
        print('input string: ', self.input_string)
        print('terms in input string matching feature vocabularies:')
        #vocabwords = []
        for feature in self.feature_dict:
            print('{}: {}'.format(feature, self.feature_dict[feature]))
        #    all_words = ' '.join(self.feature_dict[feature])
        #    vocabwords += re.split(r'[\s\,]+', all_words)
        #vocabwords = set(vocabwords)

        input_string_clean = self.feature_dict['input_string_clean']
        #input_string_clean = mapper.clean_input_string(self.input_string)
        input_word_set = set(re.split(r'[\s\,]+', input_string_clean))

        print('best match(es):')
        features = ['carrier_association','carrier_brand','carrier_name',
        'state', 'plan', 'type', 'metal']
        for i in range(self.num_rec):
            plan_words = []
            print('index in taxonomy:', self.recommendations.index[i])
            #print self.recommendations.iloc[i,:]
            plan_terms = self.recommendations[features].iloc[i,:].dropna().values
            for plan_term in plan_terms:
                plan_words += re.split(r'[\s\,]+', plan_term)
            plan_words = set(plan_words)
            intersection = plan_words & input_word_set
            missing_words = input_word_set - intersection
            print('terms in input that are unaccounted for:', missing_words, '\n\n')
        #intersection = vocabwords & input_word_set
        #missing_words = input_word_set - intersection

        #what about indiv words that do appear in taxonomy, but only in word-combinations?
        #print('terms in input that are unaccounted for:', missing_words)
