import timeit # to calculate run time
from random import randint, random

import numpy as np # for dataset creation

import scipy as sp
import scipy.stats # for pearson metric

import deap as dp # GA library
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt # plot library

def avg_pearson(individual,neighbours ):
    pearson = []
    for neighbour in neighbours:
        pearson.append((sp.stats.pearsonr(individual,neighbour)[0]+1)/2)
    return sum(pearson)/len(pearson)

start = timeit.default_timer()

#GA experiment vars
pop_size = 200 # population size
cross_p = 0.9 # crossover probability
mut_p = 0.01 # mutation probability

#GA constant vars
N = 943  # users
M = 1682  # movies
user_id = 12 # selected user id
loops = 10 # num of algorithm loops
gens = 1000 # timeout num of generations
sel_size = pop_size # selection size
tour_size = 30 # tournament size
min_delta = 0.0001 # early stopping
elite_num = 1 #number of best individuals copied to next gen (elitism)


# import data from file
data = np.genfromtxt('u.data', delimiter='\t', dtype='int')

# fill the rating dataset
dataset = np.empty((N, M))  # dataset with the ratings
dataset[:] = np.nan
for record in data:  # for every rating record in data, add a value to a cell in the dataset array
    user = int(record[0])  # get the user id
    movie = int(record[1])  # get the movie id
    rating = int(record[2])  # get the rating
    dataset[user - 1, movie - 1] = rating  # add the rating in the cell of the array with indices: (user-1,movie-1)

#choose individual user
selected_user = dataset[user_id,:].tolist()
known_ratings = np.where(np.invert(np.isnan(selected_user)))[0]

#find top-10 neighbours of user
avg_ratings = np.round(np.nanmean(dataset,0)) # raises warning that the mean of a list of nan values is nan (it is dealt with in the next line)
avg_ratings[np.isnan(avg_ratings)] = 2 # if a movie has no rating ( list of nan ), consider its mean ratings as 2 instead of nan
filled_dataset = np.array(dataset)
for i,user in enumerate(filled_dataset): #fill unknown ratings with avg rating of movie
    for j,rating in enumerate(user):
        if np.isnan(rating):
            filled_dataset[i,j] = avg_ratings[j]
pearson = []
for i,user in enumerate(filled_dataset):
    pearson.append(sp.stats.pearsonr(filled_dataset[user_id],user)[0]) #calculate correlation between selected user and users
ranking = sorted(range(len(pearson)), key=lambda i: pearson[i], reverse = True)[1:11] # top 10 users (ignore 1st because it is the selected user)
neighbours = [filled_dataset[user] for user in ranking]

# GA Algorithm loop (will run 10 times) ################################################################################

all_best_fitnesses = np.empty((loops,gens)) # list that will save for each loop a list of all the best fitnesses per generation
all_best_fitnesses[:] = np.nan
all_gens = [] # list that will save for each loop the number of generations run

#create individual class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

for k in range(loops):

    best_fitnesses = [] # list for the current algorithm loop that will save all the best fitnesses per generation

    #create initial population
    population = []
    for i in range(pop_size):
        #fill individual with the known ratings and the random ratings
        individual = np.array(selected_user)
        for j,rating in enumerate(individual):
            if np.isnan(rating):
                individual[j] = randint(1,5)
        population.append(dp.creator.Individual([int(x) for x in individual])) # create Individual object from individual list var and add to population

    #create GA model
    toolbox = dp.base.Toolbox()
    toolbox.register("evaluate", avg_pearson, neighbours=neighbours)
    toolbox.register("select", dp.tools.selTournament, population, sel_size, tour_size)
    toolbox.register("mate", dp.tools.cxOnePoint)
    toolbox.register("mutate", dp.tools.mutUniformInt, low=1, up=5, indpb=mut_p)

    #calculate initial fitness
    fitnesses = [toolbox.evaluate(individual) for individual in population]
    for individual,fitness in zip(population,fitnesses):
        individual.fitness.values = [fitness]
    best_fitness = max(fitnesses)  # best fitness of current generation
    previous_best_fitness = 0  # best fitness of previous generation

    early_stopping_flag = 0 # if early stopping has been activated
    early_stopping_patience = 5 # loops to wait before early stopping

    #run evolution
    for i in range(gens):

        print(f"Loop {k}, Generation {i}:")

        # selection
        new_population = toolbox.select() # select individuals
        new_population = [toolbox.clone(individual) for individual in new_population] # clone individuals to avoid reference to original object
        # crossover
        parents = zip(new_population[::2], new_population[1::2]) #create parent pairs from every 2 consecutive individuals
        for mom, dad in parents:
            if random() < cross_p: # apply crossover probability
                toolbox.mate(mom, dad)
                del mom.fitness.values
                del dad.fitness.values
        # mutation
        for mutant in new_population:
            toolbox.mutate(mutant)
            #repair
            for j in known_ratings:
                if mutant[j] != selected_user[j]: # if a known rating has changed
                    mutant[j] = int(selected_user[j]) # repair it
            del mutant.fitness.values

        # calculate fitness of new generation
        new_individuals = [individual for individual in new_population if not individual.fitness.valid] # get the new individuals produced (the ones with the deleted fitness)
        new_fitnesses = [toolbox.evaluate(individual) for individual in new_individuals] #calculate fitness only of the new individuals
        for individual, fitness in zip(new_individuals, new_fitnesses):
            individual.fitness.values = [fitness]

        # apply elitism (copy best N individuals from previous gen to new gen)
        elite_individuals = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x], reverse = True)[0:elite_num]
        for individual in elite_individuals:
            new_population[individual] = toolbox.clone(population[individual])

        # replace population with new population
        population[:] = new_population
        fitnesses[:] = new_fitnesses

        # save best fitness
        previous_best_fitness = best_fitness
        best_fitness = max(fitnesses)
        print(f'Best fitness: {best_fitness}')
        best_fitnesses.append(best_fitness)

        # early stopping
        if early_stopping_flag == 0: #early stopping not found
            if (abs(best_fitness - previous_best_fitness) < min_delta) | (best_fitness < previous_best_fitness):
                print("Early Stopping: Patience Activated.")
                early_stopping_flag = 1
                early_stopping_patience -= 1
        else: #early stopping found
            early_stopping_patience -= 1
            if early_stopping_patience == 0: # if patience run out
                if (abs(best_fitness - previous_best_fitness) < min_delta) | (best_fitness < previous_best_fitness): # if no fitness improvement then stop
                    print("Early Stopping: Algorithm stopped because of no fitness improvement.")
                    print()
                    early_stopping_flag = 0
                    early_stopping_patience = 5
                    break
                else: # if fitness improved reset early stopping
                    early_stopping_flag = 0
                    early_stopping_patience = 5

    # save fitnesses and gen count
    all_gens.append(i)
    best_fitnesses.extend([np.nan for i in range(gens-len(best_fitnesses))]) # elongate the vector to fit in all_best_fitnesses array
    all_best_fitnesses[k,:] = best_fitnesses

print("Genetic Algorithm finished.")
print()

# calculate general averages
avg_gen = int(np.mean(all_gens))
avg_best_fitness = np.nanmean([all_best_fitnesses[i,last_gen] for i,last_gen in enumerate(all_gens)]) # average fitness in the last generation of all algorithm loops

#calculate per generation averages for plot
avg_best_fitnesses = np.nanmean(all_best_fitnesses[:,:avg_gen+1],0)

#show results
print(f"Average best fitness: {avg_best_fitness}")
print(f"Average generations run: {avg_gen}")
plt.plot(range(avg_gen+1),avg_best_fitnesses)
plt.title(f'Pop: {pop_size}, Cross Prob: {cross_p}, Mut Prob: {mut_p}')
plt.ylabel('Pearson Metric')
plt.xlabel('Generation')
plt.show()

stop = timeit.default_timer()

print(f'Runtime: {int((stop - start)/60)} minutes')