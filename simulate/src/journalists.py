import yaml
import random
import numpy as np
from scipy import stats
from mesa import Agent

# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)

class Journalist(Agent):
    """
    Simulated journalists following the "both sides" of the debate practice
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id+20_000_000, model)
        self.agent_type = 'Journalist'
        self.story = 0.5 # first story is maximally uncertain

        # BELIEF VARIABLES
        self.belief = float(np.random.uniform(0, 1, 1)) # a bit of random journalistic bias
        self.belief_after_talk = None 
        self.belief_after_talk_media = None
        self.belief_after_talk_media_propaganda = None
        self.beliefs_encountered = []
        
        # SATISFY THE DATA COLLECTOR DURING MODEL RUNS
        self.prior = None # scientists only
        self.prior_mean = None # scientists only
        self.posterior = None # scientists only
        self.posterior_mean = None # scientists only

    def log_interaction(self, filename, list_of_interactions):
        """
        Journalists interact with science/scientists when writing stories.
        """
        with open(filename, 'a') as file:
            for scientist in list_of_interactions:
                s = f'{self.unique_id},{scientist},{self.model.schedule.steps}\n'
                file.write(s)

    def consult_science_and_write_stories(self):
        selected_beliefs = []
        # go find some science / scientists
        scientists = [a.unique_id for a in self.model.schedule.agents if a.agent_type == 'Scientist']
        scientific_beliefs = [a.belief for a in self.model.schedule.agents if a.agent_type == 'Scientist']
        # follow the norm of balance / present "both sides" by selecting the extremes (or low and median)
        belief_least_confident = min(scientific_beliefs)
        belief_least_confident_index = scientific_beliefs.index(belief_least_confident)
        selected_beliefs.append(belief_least_confident)
        self.beliefs_encountered.append(belief_least_confident)
        belief_most_confident = max(scientific_beliefs)
        belief_most_confident_index = scientific_beliefs.index(belief_most_confident)
        selected_beliefs.append(belief_most_confident)
        self.beliefs_encountered.append(belief_most_confident)
        # pick another scientist to interview at random 
        some_rando_scientist = random.choice(scientists)
        some_rando_scientist_index = scientists.index(some_rando_scientist)
        selected_beliefs.append(scientific_beliefs[some_rando_scientist_index])
        self.beliefs_encountered.append(scientific_beliefs[some_rando_scientist_index])
        # log interactions
        interacted_with = [
            scientists[belief_least_confident_index], 
            scientists[belief_most_confident_index],
            some_rando_scientist
            ]
        # add journalistic bias
        selected_beliefs.append(self.belief)
        # possibility of propaganda getting into the story
        exposure = int(stats.bernoulli(params['journalist_risk_of_exposure_to_propaganda']).rvs(1))
        if exposure == 1: # otherwise, escape unscathed
            propagandists = [p for p in self.model.schedule.agents if p.agent_type == 'Propagandist']
            propagandist = random.sample(propagandists, 1)[0]
            selected_beliefs.append(propagandist.story)
            interacted_with.append(propagandist.unique_id)
        # write story 
        self.story = float(np.mean(selected_beliefs))
        # log interactions with scientists
        self.log_interaction('../output/interactions_journalists.csv', list_of_interactions=interacted_with)
        # print(float(np.mean(selected_beliefs)))

    def step(self):
        self.consult_science_and_write_stories()