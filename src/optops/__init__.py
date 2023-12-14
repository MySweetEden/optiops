#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import datetime
import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mlflow
# mlflow.autolog()

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20

# To assure reproducibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(64)

# Create the item dictionary: item name is an integer, and value is 
# a (weight, value) 2-tuple.
items = {}
# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.uniform(0, 100))

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", random.randrange, NBR_ITEMS)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][0]
        value += items[item][1]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)

def plot_convergence_graph(gen, max):
    transposed_max = np.array(max).T
    # Create traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=gen, y=transposed_max[0],
                    mode="lines",
                    name="Fitness 1"
                ))
    fig.add_trace(go.Scatter(x=gen, y=transposed_max[1],
                    mode="lines+markers",
                    name="Fitness 2"
                ), secondary_y=True)
    # Add figure title
    fig.update_layout(
        title_text="Convergence Graph"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Generation")

    # Set y-axes titles
    fig.update_yaxes(title_text="Fitness 1", secondary_y=False)
    fig.update_yaxes(title_text="Fitness 2", secondary_y=True)
    fig.show()

def main():
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    experiment_name = "{:%Y%m%d%H%M%S}".format(now)
    experiment_id = mlflow.create_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id):
        random.seed(64)
        NGEN = 100
        MU = 50
        LAMBDA = 100
        CXPB = 0.7
        MUTPB = 0.2

        # Parameters
        mlflow.log_param('NGEN', NGEN)
        mlflow.log_param('MU', MU)
        mlflow.log_param('LAMBDA', LAMBDA)
        mlflow.log_param('CXPB', CXPB)
        mlflow.log_param('MUTPB', MUTPB)

        pop = toolbox.population(n=MU)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                halloffame=hof)
        gen, max = logbook.select("gen", "max")
        # Metrics
        mlflow.log_metric('max1', max[NGEN][0])
        mlflow.log_metric('max2', max[NGEN][1])

        plot_convergence_graph(gen, max)

if __name__ == "__main__":
    main()