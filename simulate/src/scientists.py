import os
import yaml
import random 
import numpy as np
from scipy import stats
from mesa import Agent

# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)


class BayesianScientist(Agent):
    """
    Simulated Bayesian scientists, doing research, talking to each other, and updating their beliefs. 
    You know... the grind...
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id+10_000_000, model)
        self.agent_type = 'Scientist'
        
        # SATISFY THE DATA COLLECTOR DURING MODEL RUNS
        self.story = None
        self.belief_after_talk = None
        self.belief_after_talk_media = None
        self.belief_after_talk_media_propaganda = None
        self.beliefs_encountered = None
        
        # AGENT PRIORS, PARAMETERS FOR BETA DISTRIBUTION
        alpha = int(np.random.uniform( 
            params['scientist_beta_priors_alpha_low'],
            params['scientist_beta_priors_alpha_high'],1)) 
        beta = int(np.random.uniform( 
            params['scientist_beta_priors_beta_low'],
            params['scientist_beta_priors_beta_high'],1)) 

        # All get updated as the model runs
        self.prior = [alpha, beta]
        self.prior_mean = stats.beta(*self.prior).mean()
        self.posterior = [alpha, beta]
        self.posterior_mean = stats.beta(*self.posterior).mean()
        self.discussed_belief = [alpha, beta]
        self.discussed_belief_mean = stats.beta(*self.discussed_belief).mean()
        # for convienience 
        self.belief = self.posterior_mean

        # GENERATE SAMPLE SIZES FOR AGENT RESEARCH
        sample_size_lower_bound = params['scientist_study_sample_size_lower_bound'] 
        sample_size_upper_bound = params['scientist_study_sample_size_upper_bound']
        num_sample_options = sample_size_upper_bound-sample_size_lower_bound
        sample_sizes = np.linspace(sample_size_lower_bound,sample_size_upper_bound, num_sample_options)
        self.agent_study_sample_size = int(random.choice(sample_sizes))

    def log_interaction(self, filename, alter, agent_threshold):
        with open(filename, 'a') as file:
            if self.unique_id == alter.unique_id:
                pass
            else:
                if abs(self.posterior_mean - alter.posterior_mean) < agent_threshold:
                    update = True 
                else:
                    update = False
                ij_belief_difference = self.posterior_mean - alter.posterior_mean 
                s = f'{self.unique_id},{alter.unique_id},{ij_belief_difference},{update},{self.model.schedule.steps}\n'
                file.write(s)
    
    # BAYESIAN AGENT CONDUCTS THEIR OWN RESEARCH
    ## Scientists conduct original  research and then update their priors 
     
    def agent_conducts_own_research(self, true_prob=None):
        """
        Likelihood Function: P(D|H)

        The `true_prob` argument can accept a float representing the "true" probability of the thing that
        the scientists are trying to learn. It uses that "true" probability as a parameter for the 
        Bernoulli distribution that generates their emperical evidence. 

        Alternatively, we can let the agent's prior be the parameter value for the Bernoulli distribution,
        in which case the scientists will also get back evidence that conforms to their expectations. This is
        not the behaviour we want most of the time, but can be useful for exploring some hypotheses.
        """
        self.prior = self.discussed_belief
        if true_prob:
            study_prob:float = true_prob
        else: 
            study_prob:float = stats.beta(*self.prior).rvs() # size=1 implicitly
        self.results:np.ndarray = stats.bernoulli(study_prob).rvs(size=self.agent_study_sample_size)
        self.successes = self.results.sum()
        self.failures = self.agent_study_sample_size - self.successes 

    def agent_updates_belief(self): 
        """
        Posterior Probability: P(H|D) = P(D|H) x P(H) \ P(D)
        """
        self.posterior = [
            self.prior[0] + self.successes, # alpha + successes  
            self.prior[1] + self.failures   # beta + failures
            ]
        self.posterior_mean = stats.beta(*self.posterior).mean()

    def interacts_with_other_scientists(self): 
        """
        Selects N other scientists to interact with. For each scientist, if the absolute difference
        between their mean beliefs is smaller than a given threshold, then the agent uses the other
        scientists belief to update their beliefs, and adds a record of the interaction to their
        interaction history. It uses the (signed) difference between the two agents as an edge weight.
        """
        # SELECT DISCUSSION PARTNERS
        peers = []
        for a in self.model.schedule.agents:
            if a.agent_type == 'Scientist':
                peers.append(a)
        # pick some random number of scientists to talk to, between 2 and 5
        num_discussion_partners = int(np.random.randint(2,5,1)) 
        discussion_partners = random.sample(peers, num_discussion_partners)
        # log the interaction metadata
        for partner in discussion_partners:
            self.log_interaction(filename='../output/interactions_scientists.csv', alter=partner, agent_threshold=params['scientist_difference_threshold'])

        # ASSESS CREDIBILITY, UPDATE BELIEFS
        credible_views = []
        for partner in discussion_partners:
            if abs(self.posterior_mean - partner.posterior_mean) < params['scientist_difference_threshold']:
                credible_views.append(partner.posterior)
        # UPDATE BELIEFS IF THE SOURCE IS CREDIBLE
        if len(credible_views) > 0:
            for v in credible_views:
                # update agent posterior and posterior mean
                self.posterior = [
                    # sum of alpha parameters for self prior and peer posterior
                    self.prior[0] + v[0], 
                    # sum of beta parameters for self prior and peer posterior
                    self.prior[1] + v[1] 
                ]
                self.posterior_mean = stats.beta(*self.posterior).mean()

                # update agent discussed belief and discussed belief mean
                self.discussed_belief = [
                    # sum of alpha parameters for self prior and peer posterior
                    self.prior[0] + v[0], 
                    # sum of beta parameters for self prior and peer posterior
                    self.prior[1] + v[1] 
                ]
                self.discussed_belief_mean = stats.beta(*self.discussed_belief).mean()

    def step(self): 
        self.agent_conducts_own_research(true_prob=params['scientist_research_bernoulli_probability'])
        self.agent_updates_belief()
        self.interacts_with_other_scientists()