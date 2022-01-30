import yaml
import random 
import numpy as np
from mesa import Agent

# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)

class Policymaker(Agent):
    """
    Simulated policymaker trying to make policy...
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id+50_000_000, model)
        self.agent_type = 'Policymaker'

        # BELIEF VARIABLES
        self.belief = float(np.random.uniform(0, 1, 1)) # some population heterogeneity in initial beliefs
        self.belief_after_talk = None  
        self.belief_after_talk_media = None
        self.belief_after_talk_media_propaganda = None
        self.beliefs_encountered = []

        # SATISFY THE DATA COLLECTOR DURING MODEL RUNS
        self.prior = None
        self.prior_mean = None
        self.posterior = None
        self.posterior_mean = None
        self.story = None
        
    def log_interaction(self, filename, alter, agent_threshold):
        with open(filename, 'a') as file:
            if abs(self.belief - alter.belief) < agent_threshold:
                update = True 
            else:
                update = False
            ij_belief_difference = self.belief - alter.belief 
            s = f'{self.unique_id},{alter.unique_id},{ij_belief_difference},{update},{self.model.schedule.steps}\n'
            file.write(s)

    def interaction(self):
        peers = []
        for a in self.model.schedule.agents:
            if a.agent_type == 'Policymaker':
                peers.append(a)
        # select N discussion partners, between 2 and 5
        num_discussion_partners = int(np.random.randint(2,5,1))
        discussion_partners = random.sample(peers, num_discussion_partners)
        # the interaction
        for partner in discussion_partners:
            # log the encountered belief
            self.beliefs_encountered.append(partner.belief)
            # updates belief or not
            if abs(self.belief - partner.belief) < params['policymaker_difference_threshold']:
                self.belief_after_talk = int(np.mean([self.belief, partner.belief]))
                self.belief = self.belief_after_talk
            else:
                self.belief_after_talk = self.belief
            # log the interaction metadata
            self.log_interaction(filename='../output/interactions_policymakers.csv', alter=partner, agent_threshold=params['policymaker_difference_threshold'])

    def consumes_news_media(self):
        # reading_options = []
        journalists = [j.story for j in self.model.schedule.agents if j.agent_type == 'Journalist']
        # the agent will read anywhere between 2 and 10 news stories
        number_of_stories_to_read =  int(np.random.randint(2, 10, 1)) 
        # sample stories to read
        to_read = random.sample(journalists, number_of_stories_to_read)
        average_belief_in_stories = np.mean(to_read)
        weighted_opinion = np.average([self.belief_after_talk, average_belief_in_stories], weights=[0.7, 0.2]) # beliefs from the policy community matter more than journalists...
        self.belief_after_talk_media = weighted_opinion
        self.belief = self.belief_after_talk_media
        
        
    def encounters_propaganda(self):
        """
        All policymakers encounter propaganda, the question is how much...
        """
        propagandists = [p.story for p in self.model.schedule.agents if p.agent_type == 'Propagandist']
        # the agent will read anywhere between 5 and 10 pieces of bullshit
        amount_of_bullshit_to_read = int(np.random.randint(5, 10, 1)) 
        # sample bullshit to read
        to_read = random.sample(propagandists, amount_of_bullshit_to_read)
        to_read.append(self.belief_after_talk_media)
        average_belief_in_bullshit = np.mean(to_read)
        weighted_opinion = np.average([self.belief_after_talk_media, average_belief_in_bullshit], weights=[0.8, 0.5]) # they still know some bullshit when they see it, so don't take it too seriously
        self.belief_after_talk_media_propagandist = weighted_opinion 
        self.belief = self.belief_after_talk_media_propagandist

    def step(self):
        self.interaction()
        self.consumes_news_media()
        self.encounters_propaganda()