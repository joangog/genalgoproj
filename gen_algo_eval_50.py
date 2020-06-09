import timeit # to calculate run time

from random import randint, random
import operator # for max() to return value AND index

import numpy as np # for dataset creation

import scipy as sp
import scipy.stats # for pearson metric

from sklearn.metrics import mean_squared_error # for RMSE and MSE

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
pop_size = 2 # population size
cross_p = 0.9 # crossover probability
mut_p = 0.01 # mutation probability

#GA constant vars
N = 943  # users
M = 1682  # movies
user_ids = [x for x in range(0,50)] # selected user ids
loops = 10 # num of algorithm loops
gens = 1000 # timeout num of generations
sel_size = pop_size # selection size
tour_size = 30 # tournament size
min_delta = 0.001 # early stopping
elite_num = 1 #number of best individuals copied to next gen (elitism)


# import data from file
train_data = np.genfromtxt('ua.base', delimiter='\t', dtype='int')
test_data = np.genfromtxt('ua.test', delimiter='\t', dtype='int')

# fill the train dataset
train_dataset = np.empty((N, M))  # dataset with the ratings
train_dataset[:] = np.nan
for record in train_data:  # for every rating record in data, add a value to a cell in the dataset array
    user = int(record[0])  # get the user id
    movie = int(record[1])  # get the movie id
    rating = int(record[2])  # get the rating
    train_dataset[user - 1, movie - 1] = rating  # add the rating in the cell of the array with indices: (user-1,movie-1)

# fill the test dataset
test_dataset = np.empty((N, M))  # dataset with the ratings
test_dataset[:] = np.nan
for record in test_data:  # for every rating record in data, add a value to a cell in the dataset array
    user = int(record[0])  # get the user id
    movie = int(record[1])  # get the movie id
    rating = int(record[2])  # get the rating
    test_dataset[user - 1, movie - 1] = rating  # add the rating in the cell of the array with indices: (user-1,movie-1)

#create individual class
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


# The algorithm will run for 50 different users

all_best_fitnesses = np.empty((len(user_ids),loops, gens))  # list that will save for each user, for each algo loop, a list of all the best fitnesses per generation
all_best_fitnesses[:] = np.nan
all_best_individuals = np.empty((len(user_ids),loops, gens,M))  # list that will save for each user, for each algo loop, a list of all the best individuals (M sized lists) per generation
all_best_individuals[:] = np.nan
all_gens = np.empty((len(user_ids),loops),dtype='int')  # list that will save for each loop the number of generations run

for user_id in user_ids:

    #choose individual user
    selected_user = train_dataset[user_id,:].tolist()
    known_ratings = np.where(np.invert(np.isnan(selected_user)))[0]

    #find top-10 neighbours of user
    avg_ratings = np.round(np.nanmean(train_dataset,0)) # raises warning that the mean of a list of nan values is nan (it is dealt with in the next line)
    avg_ratings[np.isnan(avg_ratings)] = 2 # if a movie has no rating ( list of nan ), consider its mean ratings as 2 instead of nan

    filled_dataset = np.array(train_dataset)
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

    for k in range(loops):

        best_fitnesses = [] # list for the current algorithm loop that will save all the best fitnesses per generation
        best_individuals = np.empty((gens,M)) # list for the current algorithm loop that will save all the best individuals (M sized list) per generation
        best_individuals[:] = np.nan
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

            print(f"User {user_id}, Loop {k}, Generation {i}:")

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

            # save best individual and fitness
            previous_best_fitness = best_fitness
            index,best_fitness = max(enumerate(fitnesses),key=operator.itemgetter(1)) #save best fitness and save the index too
            print(f'Best fitness: {best_fitness}')
            best_fitnesses.append(best_fitness)
            best_individuals[i] = population[index]

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

        # save gen count, fitnesses and best individuals per gen
        all_gens[user_id,k]=i
        best_fitnesses.extend([np.nan for i in range(gens-len(best_fitnesses))]) # elongate the vector to fit in all_best_fitnesses array
        all_best_fitnesses[user_id,k,:] = best_fitnesses
        all_best_individuals[user_id,k,:] = best_individuals

print("Genetic Algorithm finished.")
print()

#calculate rsme and mse comparing the predicted to the true values of the test dataset
all_mse_vals = np.empty((len(user_ids),loops,gens))
all_mse_vals[:] = np.nan
all_rmse_vals = np.empty((len(user_ids),loops,gens))
all_rmse_vals[:] = np.nan
for user_id in user_ids:
    for i,loop_best_individuals in enumerate(all_best_individuals[user_id,:,:]):
        for j,best_individual in enumerate(loop_best_individuals):
            if not any(np.isnan(best_individual)):
                true_ratings = test_dataset[user_id,np.invert(np.isnan(test_dataset[user_id]))]
                predict_ratings = best_individual[np.invert(np.isnan(test_dataset[user_id]))]
                mse = mean_squared_error(true_ratings,predict_ratings)
                rmse = np.sqrt(mse)
                all_mse_vals[user_id,i,j] = mse
                all_rmse_vals[user_id,i,j] = rmse

# calculate average gens run of all algorithm loops of all users
avg_gen = int(np.mean(all_gens))

#calculate average fitness in the last generation of all algorithm loops of all users
all_best_fitnesses_list = []
for user in range(len(user_ids)):
    for i,last_gen in enumerate(all_gens[user]):
        k = all_best_fitnesses[user,i,last_gen]
        all_best_fitnesses_list.append(all_best_fitnesses[user,i,last_gen])
avg_best_fitness = np.mean(all_best_fitnesses_list)

#calculate average mse in the last generation of all algorithm loops of all users
all_mse_vals_list = []
for user in range(len(user_ids)):
    for i,last_gen in enumerate(all_gens[user]):
        all_mse_vals_list.append(all_mse_vals[user,i,last_gen])
avg_mse = np.mean(all_mse_vals_list)

#calculate average rmse in the last generation of all algorithm loops of all users
all_rmse_vals_list = []
for user in range(len(user_ids)):
    for i,last_gen in enumerate(all_gens[user]):
        all_rmse_vals_list.append(all_rmse_vals[user,i,last_gen])
avg_rmse = np.mean(all_rmse_vals_list)

#calculate per generation averages for plot
avg_best_fitnesses = np.nanmean(all_best_fitnesses[:,:,:avg_gen+1],(0,1))
avg_mse_vals = np.nanmean(all_mse_vals[:,:,:avg_gen+1],(0,1))
avg_rmse_vals = np.nanmean(all_rmse_vals[:,:,:avg_gen+1],(0,1))

#show results
print(f"Average best fitness: {avg_best_fitness}")
print(f"Average generations run: {avg_gen}")
print(f"Average MSE: {avg_mse}")
print(f"Average RMSE: {avg_rmse}")
# Pearson Metric Plot
plt.figure(1)
plt.plot(range(avg_gen+1),avg_best_fitnesses)
plt.title(f'Pop: {pop_size}, Cross Prob: {cross_p}, Mut Prob: {mut_p}')
plt.ylabel('Pearson Metric')
plt.xlabel('Generation')
plt.show()
# MSE Plot
plt.figure(2)
plt.plot(range(avg_gen+1),avg_mse_vals)
plt.title(f'Pop: {pop_size}, Cross Prob: {cross_p}, Mut Prob: {mut_p}')
plt.ylabel('MSE')
plt.xlabel('Generation')
plt.show()
# RMSE Plot
plt.figure(3)
plt.plot(range(avg_gen+1),avg_rmse_vals)
plt.title(f'Pop: {pop_size}, Cross Prob: {cross_p}, Mut Prob: {mut_p}')
plt.ylabel('RMSE')
plt.xlabel('Generation')
plt.show()

stop = timeit.default_timer()

print(f'Runtime: {int((stop - start)/60)} minutes')