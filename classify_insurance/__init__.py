"""
Package for mapping strings describing insurance plans into Ribbon Health taxonomy.
This package was built for a 3-week consulting project with Ribbon Health (ribbonhealth.com)
as part of the Insight Data Science program.

Modules:
planmapper.py
Defines PlanComparisonModel class.
A model (i.e. an instance of this class) performs comparison of input text string to insurance plans in taxonomy.

inputwithmappings.py
Defines InputWithMappings class.
An instance of this class contains an input string and information on how it compares with plans in the taxonomy.

simvec2score.py
Defines SimilarityVectorToScore class.
An instance of this class maps a similarity vector (between an input string and an insurance plan) to a similarity score.
"""
