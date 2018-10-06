"""
Example script for using classify_insurance package to take a fuzzy input string
describing an insurance plan and recommending matching plans in Ribbon Health taxonomy.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from classify_insurance.planmapper import PlanComparisonModel
from classify_insurance.inputwithmappings import InputWithMappings


def get_mapped_input(mapper, ix = None):
    """Select an already mapped input string and show properties of string and matching plans.

    Arguments:
        mapper (PlanComparisonModel obj): model for matching inputs to taxonomy.
        ix (int): index of input string in confirmed mappings df. If None, select random index.
    Returns:
        input_string (str): input text string.
        UUID_true (str): matching plan UUID.
    """

    if not mapper.mapped_plans_loaded:
        mapper.load_mapped_plans()
    mapped_plans = mapper.mapped_plans
    if not ix:
        ix = np.random.choice(mapped_plans.index.tolist())

    input_mapped_plan = mapper.index_to_mapped_input_plan(ix, mapped_plans)
    input_string = input_mapped_plan['input_string']

    print('\nINPUT:')
    print('index {}:'.format(ix))
    print('input string: "{}"; Cleaned: "{}"'.format(input_string, mapper.clean_input_string(input_string)))
    input_feature_dict = input_mapped_plan['input_feature_dict']
    print('Features of input string:')
    for feature in mapper.features:
        print('{}: {}'.format(feature, input_feature_dict[feature]))

    print('\nTRUE MATCHING PLAN:')
    UUID_true = input_mapped_plan['true_plan']['UUID']
    print('Unique ID: {}'.format(UUID_true))
    true_plan_feature_dict = input_mapped_plan['true_plan']
    print('Features of true plan:')
    for feature in mapper.features:
        print('{}: {}'.format(feature, true_plan_feature_dict[feature]))

    sim_vec_true = mapper.similarity_vector(input_feature_dict, true_plan_feature_dict)
    sim_score_true = mapper.sim_vec_to_score.compute_score(sim_vec_true)
    print('similarity vector:', sim_vec_true)
    print('score:', sim_score_true)
    print('original plan description before cleaning:')
    print(mapper.taxonomy_orig[mapper.taxonomy_orig['Ribbon Insurance UUID'] == UUID_true].to_dict(orient='records'))
    return (input_string, UUID_true)

def get_plan_recommendations(input_string, mapper, margin = 0.2):
    """Find recommended mathing plans to input string.

    Arguments:
        input_string (str): input text string.
        mapper (PlanComparisonModel obj): model for matching inputs to taxonomy.
        margin (flt): recommend all plans within margin of top similarity score.
    Returns:
        plan_info (InputWithMappings obj): contains recommended matches information.
    """
    print('\nCOMPARING INPUT TO INSURANCE PLANS...')
    plan_info = InputWithMappings(input_string)
    plan_info.compute_similarity_vectors_scores(mapper)
    plan_info.get_recommendations(margin = 0.2, verbose=True)
    return plan_info

def main():
    args = sys.argv[1:]

    if len(args[:]) != 0:
        print('PROGRAM TAKES ZERO ARGUMENTS. EXITING.')
        sys.exit(1)

    ########################################################
    #load the model that maps input strings to insurance plans:
    mapper = PlanComparisonModel()

    ########################################################
    #Choose from 2 options below:

    #1. use an already mapped input string (and print lots of information):
    input_string, UUID_true = get_mapped_input(mapper, ix = 116)
    #2. or directly select your own input string:
    #input_string = 'bcbs - anthem pathway x ind ppo direct access ct'
    #UUID_true = None

    ########################################################
    #given the input string only, find recommended matches in the plan taxonomy:
    plan_info = get_plan_recommendations(input_string, mapper)

    print('\nTHE BEST MATCHING PLAN(S):')
    for i_recommended in range(plan_info.num_rec):
        print('Rank {}: {}'.format(i_recommended + 1, plan_info.recommended_plan_dicts[i_recommended]))
    if not plan_info.good_matches_found:
        plan_info.recommend_taxonomy_entry()

    #if you started with an already mapped plan, you can check here (again)
    #what is the true matching plan:
    if isinstance(UUID_true, str):
        if UUID_true in plan_info.recommendations['UUID'].values:
            print('KNOWN TRUE MATCHING PLAN:')
            print(plan_info.recommendations[plan_info.recommendations['UUID'] == UUID_true].to_dict(orient='records'))
    print('\n')

if __name__ == '__main__':
    main()
