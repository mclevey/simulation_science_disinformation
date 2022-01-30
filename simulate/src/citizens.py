import yaml
import random 
import numpy as np
from mesa import Agent
from scipy import stats

# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)

class Citizen(Agent):
    """
    Simulated citizen trying to figure out what the hell is going on...
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id+40_000_000, model)
        self.agent_type = 'Citizen'
        
        # BELIEF VARIABLES
        self.belief = float(np.random.uniform(0, 1, 1)) # population heterogeneity
        self.belief_after_talk = None
        self.belief_after_talk_media = None
        self.belief_after_talk_media_propaganda = None 
        self.beliefs_encountered = []

        # SATISFY THE DATA COLLECTOR 
        self.prior = None # scientists only
        self.prior_mean = None # scientists only
        self.posterior = None # scientists only
        self.posterior_mean = None # scientists only
        self.story = None # journalists and propagandists only

    def log_interaction(self, filename, alter, agent_threshold):
        with open(filename, 'a') as file:
            if self.unique_id == alter.unique_id:
                pass # ignore self loops entirely
            else:
                if abs(self.belief - alter.belief) < agent_threshold:
                    update = True 
                else:
                    update = False
                ij_belief_difference = self.belief - alter.belief 
                s = f'{self.unique_id},{alter.unique_id},{ij_belief_difference},{update},{self.model.schedule.steps}\n'
                file.write(s)

    def log_movements(self, filename):
        with open(filename, 'a') as file:
            s = f'{self.unique_id},{self.pos[0]},{self.pos[1]},{self.model.schedule.steps}\n'
            file.write(s)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=params['moore'],
            include_center = params['include_center'])
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        self.log_movements('../output/travel_citizens.csv')
    
    def interaction(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        peers = []
        if len(cellmates) > 1:
            for c in cellmates:
                if c.agent_type == 'Citizen':
                    peers.append(c)
            # select N discussion partners
            if len(peers) > 1: 
                if len(peers) < 10:
                    num_discussion_partners = int(np.random.randint(1,len(peers),1)) 
                else: # don't let them talk to 500 people in one step... 
                    num_discussion_partners = int(np.random.randint(1,10,1)) 
                discussion_partners = random.sample(peers, num_discussion_partners)
                # the interaction
                for partner in discussion_partners:
                    # log the encountered belief
                    self.beliefs_encountered.append(partner.belief)
                    # update beliefs or not
                    if abs(self.belief - partner.belief) < params['citizen_difference_threshold']:
                        self.belief_after_talk = int(np.mean([self.belief, partner.belief]))
                        self.belief = self.belief_after_talk
                    else:
                        self.belief_after_talk = self.belief
                    # log the interaction metadata
                    self.log_interaction(filename='../output/interactions_citizens.csv', alter=partner, agent_threshold=params['citizen_difference_threshold'])

    def consumes_news_media(self):
        journalists = [j.story for j in self.model.schedule.agents if j.agent_type == 'Journalist']
        # the agent will read anywhere between 2 and 10 news stories by journalists
        number_of_stories_to_read = int(np.random.randint(2, 10, 1)) 
        # sample stories to read
        stories = random.sample(journalists, number_of_stories_to_read)
        # log the encountered belief
        for s in stories: # 'stories' are just floats representing the 'belief' the story presents
            self.beliefs_encountered.append(s)
        try:
            # will fail if no discussion partners have been encountered yet, or nothing to update inital None assignment
            self.belief_after_talk_media = np.average([self.belief_after_talk, s], weights=[0.6, 0.3]) 
            # self.belief_after_talk_media = np.mean([self.belief_after_talk, s])
        except:
            self.belief_after_talk_media = np.average([self.belief, s], weights=[0.6, 0.3]) 
        self.belief = self.belief_after_talk_media
        
    def encounters_propaganda(self):
        # do they get exposed to propaganda?
        exposure = int(stats.bernoulli(.8).rvs(1))
        if exposure == 1: # otherwise, escape unscathed
            propagandists = [p.story for p in self.model.schedule.agents if p.agent_type == 'Propagandist']
            # the agent will read anywhere between 2 and 10 pieces of bullshit
            amount_of_bullshit_to_read =  int(np.random.randint(2, 10, 1)) 
            # sample bullshit to read
            stories = random.sample(propagandists, amount_of_bullshit_to_read)
            for s in stories:
                self.beliefs_encountered.append(s)
                self.belief_after_talk_media_propaganda = np.average([self.belief_after_talk_media, s], weights=[0.6, 0.3])
            self.belief = self.belief_after_talk_media

    def step(self):
        self.move()
        self.interaction()
        self.consumes_news_media()
        self.encounters_propaganda()