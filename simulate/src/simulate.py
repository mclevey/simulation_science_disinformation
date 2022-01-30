import os
import yaml
import numpy as np
import pandas as pd
from model import World
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%B %d (%Y) @ %H:%M:%S")

# Prepare files for interaction logging (including clearing results from previous runs)
log = ['scientists', 'journalists', 'policymakers', 'citizens', 'propagandists']
for agent_type in log:
    file = f'../output/interactions_{agent_type}.csv'
    try:
        os.remove(file)
        with open(file, 'w') as file:
            if agent_type == 'journalists':
                file.write('i,j,step\n')
            else:
                file.write('i,j,ij_belief_difference,update_boolean,step\n')
    except:
        pass


# Prepare files for travel logging (including clearing results from previous runs)
log = ['scientists', 'journalists', 'policymakers', 'citizens', 'propagandists']
for agent_type in log:
    file = f'../output/travel_{agent_type}.csv'
    try:
        os.remove(file)
        with open(file, 'w') as file:
            file.write('agent,x,y,step\n')
    except:
        pass


# LOAD MODEL PARAMETERS
with open(r'../input/parameters.yaml') as params:
    params = yaml.load(params, Loader=yaml.FullLoader)

def print_select_model_parameters():
    os.system('clear')
    print('Run: ', date_time, '\n')
    print('####################')
    print('# MODEL PARAMETERS #')
    print('####################')
    print('\n')
    print('Number of Citizens:', params['num_citizens'])
    print('Number of Scientists:', params['num_scientists'])
    print('Number of Journalists:', params['num_journalists'])
    print('Number of Propagandists:', params['num_propagandists'])
    print(f'Number of Simulations:', params['number_of_simulations'])
    print(f'Number of Steps in Each Simulation:', params['steps_per_model'])
    print('\n')

print_select_model_parameters()

# EXECUTE SIMULATIONS
result_dfs = []

print('###################')
print('### SIMULATIONS ###')
print('###################')
print('\n')

for i in range(params['number_of_simulations']):
    print('Executing Simulation', i)
    
    run = World(
        num_scientists = params['num_scientists'],
        num_citizens = params['num_citizens'],
        num_journalists = params['num_journalists'],
        num_propagandists = params['num_propagandists'],
        num_policymakers=params['num_policymakers'],
        width = 10,
        height = 10
    )

    for j in range(params['steps_per_model']):
        run.step()
    
    agent_beliefs = run.datacollector.get_agent_vars_dataframe().reset_index()
    agent_beliefs['SimulationID'] = i
    result_dfs.append(agent_beliefs)

    df = pd.concat(result_dfs)  

print('\n')
print('################')
print('### FINISHED ###')
print('################')

# print('\n')
# print(df.info())

# STORE RESULTS
df.to_csv('../output/model_runs.csv', index=False)