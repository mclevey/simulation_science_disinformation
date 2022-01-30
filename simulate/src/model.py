from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from scientists import BayesianScientist
from propagandists import Propagandist
from citizens import Citizen
from journalists import Journalist
from policymakers import Policymaker


class World(Model):
    """The model our agents live in."""
    def __init__(self, num_scientists, num_citizens, num_journalists, num_propagandists, num_policymakers, width, height):
        
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        #####################
        ### CREATE AGENTS ###
        #####################

        # Bayesian Scientists
        for scientist_i in range(num_scientists):
            scientist = BayesianScientist(scientist_i, self)
            self.schedule.add(scientist)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(scientist, (x, y))

        # Journalists
        for journalist_i in range(num_journalists):
            journalist = Journalist(journalist_i, self)
            self.schedule.add(journalist)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(journalist, (x, y))

        # Propagandists
        for propagandist_i in range(num_propagandists):
            propagandist = Propagandist(propagandist_i, self)
            self.schedule.add(propagandist)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(propagandist, (x, y))

        # Citizens
        for citizen_i in range(num_citizens):
            citizen = Citizen(citizen_i, self)
            self.schedule.add(citizen)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(citizen, (x, y))

        # Policymakers
        for policymaker_i in range(num_policymakers):
            policymaker = Policymaker(policymaker_i, self)
            self.schedule.add(policymaker)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(policymaker, (x, y))

        #####################################
        #### COLLECT DATA FROM MODEL RUNS ###
        #####################################

        self.datacollector = DataCollector(
            agent_reporters={
                'Agent Type':'agent_type',
                "Prior":"prior", 
                "Prior Mean":"prior_mean",
                "Posterior":'posterior', 
                "Posterior Mean":"posterior_mean",
                'Story':'story',
                'Belief (After All Step Actions)':'belief',
                'Belief After Talk':'belief_after_talk',
                'Belief After Talk Media':'belief_after_talk_media',
                'Belief After Talk Media Propaganda':'belief_after_talk_media_propaganda',
                'Beliefs Encountered':'beliefs_encountered'
                }
            )

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()