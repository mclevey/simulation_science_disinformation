import yaml
import numpy as np
from mesa import Agent

# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)

class Propagandist(Agent):
    """
    Simulated propagandist cherry picks studies from the low end of the belief distribution.
    They also "write stories" that people can read / listen to / watch / whatever.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id+30_000_000, model)
        self.agent_type = 'Propagandist'

        # BELIEF VARIABLES
        self.belief = float(np.random.uniform(0, 0.2, 1)) # their ideological bias
        self.story = self.belief 

        # SATISFY THE DATA COLLECTOR DURING MODEL RUNS
        self.prior = None # scientists only
        self.prior_mean = None # scientists only
        self.posterior = None # scientists only
        self.posterior_mean = None # scientists only
        self.belief_after_talk = None # citizens and policymakers only
        self.belief_after_talk_media = None # citizens and policymakers only
        self.belief_after_talk_media_propaganda = None # citizens and policymakers only
        self.beliefs_encountered = None # all give, no take
        
    # PROPAGANDIST MAKES PROPAGANDA
    def propogandist_writes_propaganda(self):
        """
        They selectively report studies that are against the scientific consensus.
        They add their own negative framing (belief < 0.5) with variable intensity (drawn from uniform dist)
        """
        # biased, duh!
        propagandist_bias = self.belief
        # all the science to cherry pick
        science_options = []
        # cherry pick the science
        for a in self.model.schedule.agents:
            if a.agent_type == 'Scientist':
                science_options.append(a.posterior_mean)
        low_confidence_science = min(science_options)
        weighted_bias = np.average([propagandist_bias, low_confidence_science], weights=[0.8, 0.4])
        self.story = float(weighted_bias)

    def step(self):
        self.propogandist_writes_propaganda()
        