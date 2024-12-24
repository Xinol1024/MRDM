import pandas as pd
import numpy as np

from RL_utils.scoring_function import MinMaxGaussianModifier, ArithmeticMeanScoringFunction, ExponentialRangeScoreModifier
from RL_utils.custom_scoring_fcn import QED, SAScorer, LigandEfficancy, LogP, MW


def build_scoring_function(scoring_definition, 
                           fscores, 
                           return_individual = False):
    """ Build scoring function """

    # scoring definition has columns:
    # category, name, minimize, mu, sigma, file, model, n_top
    df = pd.read_csv(scoring_definition, sep=",",header=0)
    scorers = {}

    for i, row in df.iterrows():
        name = row['name']


        if row.category == "qed":
            scorers[name] = QED(score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize))
        elif row.category == "sa":
            scorers[name] = SAScorer( 
                                    score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    fscores=fscores  
                                    )

        elif row.category == 'ligand_efficiency':
            scorers[name] = LigandEfficancy(
                                    score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    model_path=row.file
                                    )         
        elif row.category == 'logp':
            scorers[name] = LogP(
                                    score_modifier=ExponentialRangeScoreModifier( lower_bound=row.lower_bound, 
                                                                                 upper_bound=row.upper_bound, 
                                                                                 decay_rate=0.5),
                                    )     
        elif row.category == 'mw':
            scorers[name] = MW(
                                    score_modifier=ExponentialRangeScoreModifier( lower_bound=row.lower_bound, 
                                                                                 upper_bound=row.upper_bound, 
                                                                                 decay_rate=0.5),
                                    )                
        else:
            print("WTF Did not understand category: {}".format(row.category))
 
    scoring_function = ArithmeticMeanScoringFunction([scorers[i] for i in scorers])

    if return_individual:
        return scorers, scoring_function
    else:
        return scoring_function