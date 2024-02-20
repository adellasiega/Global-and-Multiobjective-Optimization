import random 
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ipywidgets as widgets
from ipywidgets import interact

class GeneticBrickSolver():
    
    def __init__(self, population_size, mutation_rate, crossover_rate, max_generations, tournament_size, k_elitism):
        
        # Parameters of the genetic algorithm
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.k_elitism = k_elitism        

        # Population of the algorithm
        self.population = []

        # Results of the algorithm
        self.best_individual = None
        self.best_fitness = np.infty  
        self.history = [] # list of the best individual and its fitness for each generation
    
        # Specs of the problem
        self.columns_per_individual = None
        self.bricks_per_column = None
        self.brick_heights = None
    
    def generate_random_individual(self):
        '''
            Method to randomly generate an individual.
            An individual is a fesible solution of the problem. 
            In this case, an individual is a list of columns,
            where each column is a list of bricks.
        '''
        individual = []
        mn = self.bricks_per_column * self.columns_per_individual
        random_bricks = random.sample(range(mn), k=mn)
        for i in range(self.columns_per_individual):
            column = random_bricks[i*self.bricks_per_column:(i+1)*self.bricks_per_column]
            individual.append(column)
        return individual

    def initialize_population(self):
        '''
            Method to randomly generate the initial population.
        '''
        initial_population = [self.generate_random_individual() for _ in range(self.population_size)]
        self.population = copy.deepcopy(initial_population)

    def column_height(self, column):
        '''
            Method to calculate the height of a column.
            The height of a column is the sum of the heights of each brick.
        '''
        H = sum(self.brick_heights[brick] for brick in column)
        return H

    def fitness(self, individual):
        '''
            Method to calculate the fitness of an individual.
            The fitness of an individual is the difference 
            between its highest column and itsdio shortest one
        '''
        column_min = min(individual, key=self.column_height)
        column_max = max(individual, key=self.column_height)
        hight_min = self.column_height(column_min)
        hight_max = self.column_height(column_max)
        return hight_max - hight_min
        
    def update_best(self, generation):
        '''
            Update the best individual and its fitness
        '''
        for individual in self.population:
            current_fitness = self.fitness(individual)
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_individual = copy.deepcopy(individual)

        record = (generation, self.population, self.best_individual, self.best_fitness)
        self.history.append(record)
                
    def elitism(self):
        '''
            Method to perform elitism. Returns a list containig 
            the k_elitism best individuals of the population, they
            are removed from the population.
        '''
        if self.is_odd(self.population_size) ^ self.is_odd(self.k_elitism):
            self.k_elitism += 1
        
        self.population.sort(key=self.fitness)
        elites = copy.deepcopy(self.population[:self.k_elitism])
        self.population = copy.deepcopy(self.population[self.k_elitism:])
        random.shuffle(self.population)
        return elites
    
    def tournament_selection(self):
        '''
            Method to perform a tournament selection.
            The tournament selection is performed by randomly selecting 
            some individuals from the population and returning the best one.
        '''
        tournament = random.sample(self.population, k=self.tournament_size)
        best_individual = tournament[0]
        for individual in tournament:
            if self.fitness(individual) < self.fitness(best_individual):
                best_individual = individual
        return best_individual

    def crossover(self, parent1, parent2):
        '''
            Method to perform a crossover between two parents.
            The crossover is performed by randomly selecting a column on parent1
            and a column on parent2. The two columns are swapped.
            It is needed to restore the condition that every individual has just
            one brick of each type
        '''
        rand_idx = random.randint(0, self.columns_per_individual-1)
        column1 = parent1[rand_idx].copy()
        column2 = parent2[rand_idx].copy()
        
        #Create the mapping
        x = list(set(column1) - set(column2))
        y = list(set(column2) - set(column1))

        mapping = dict(zip(x, y))

        #Create the children
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        #Swap the columns
        child1[rand_idx] = column2.copy()
        child2[rand_idx] = column1.copy()

        #Restore the condition that every individual has just one brick of each type
        for col_idx in range(self.columns_per_individual):
            if col_idx != rand_idx:
                for key, value in mapping.items():
                    child1[col_idx] = list(np.where(child1[col_idx] == value, key, child1[col_idx]))
                    child2[col_idx] = list(np.where(child2[col_idx] == key, value, child2[col_idx]))
        
        return child1, child2
    
    def mutation(self, individual):
        '''
            Method to perform a mutation.
            The mutation is performed by randomly selecting two columns,
            then selecting two random bricks and then swapping them.
        '''
        rc1, rc2 = random.sample(range(self.columns_per_individual), k=2)
        rb1, rb2 = random.choices(range(self.bricks_per_column), k=2)
        individual[rc1][rb1], individual[rc2][rb2] = individual[rc2][rb2], individual[rc1][rb1]
        return individual
    
    def is_odd(self, number):
        '''
            Method to check if a number is odd.
        '''
        return number & 1

    def check_individual(self, individual):
        '''
            Method to check if an individual is feasible.
            An individual is feasible if it has just one brick of each type.
        '''
        bricks = []
        for column in individual:
            for brick in column:
                if brick in bricks:
                    return False
                bricks.append(brick)
        return True
    
    def generate_new_population(self):
        '''
            Method to generate a new population.
            The new population is generated by following these steps:
                1. elitism
                2. tournament selection
                3. crossover
                4. mutation
        '''
        new_population = []

        #Elitism
        new_population += self.elitism()

        #Tournament selection
        selected = [self.tournament_selection() for _ in range(len(self.population))]

        #Genetic operators
        parent_pairs = [random.sample(selected, k=2) for _ in range(len(self.population)//2)]
        
        for parent1, parent2 in parent_pairs:
            child1, child2 = parent1, parent2
            #Crossover step
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            #Mutation step
            if random.random() < self.mutation_rate:
                child1 = self.mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = self.mutation(child2)
                
            new_population.append(child1)
            new_population.append(child2)

        #Update population
        
        self.population = copy.deepcopy(new_population)
    
    def solve(self, brick_heights, columns_per_individual, bricks_per_column):
        '''
            Method to solve the problem.
        '''
        self.columns_per_individual = columns_per_individual
        self.bricks_per_column = bricks_per_column
        self.brick_heights = brick_heights

        self.initialize_population()
        self.update_best(0)        
        for i in range(self.max_generations):
            #print a string to show the progress of the algorithm
            print(f'Generation {i+1}/{self.max_generations} ({100*(i+1)/self.max_generations}%)', end='\r')
            self.generate_new_population()
            self.update_best(i+1)
    
    def plot_fitness(self):
        '''
            Method to plot the best fitness for each generation.
        '''
        generations = [record[0] for record in self.history]
        fitnesses = [record[3] for record in self.history]
        plt.plot(generations, fitnesses)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def plot_population(self):
        '''
            Method to plot each individual of the population using
            a slider to explore the generations. Use this function
            only if the number of columns and individuals is small.
        '''
        def print_results(generation):
            _,population,_,_ = self.history[generation]
            fig, ax = plt.subplots(1, len(population), figsize=(20, 5))
            fig.suptitle(f'Generation {generation}')
            for idx_individual, individual in enumerate(population):
                ax[idx_individual].set_title(f"Sol: {idx_individual}, f = {self.fitness(individual):.1f}")
                for idx_col in range(len(individual)):
                    column_height = 0
                    for idx_bri in range(len(individual[idx_col])):
                        current_brick = individual[idx_col][idx_bri]
                        current_height = self.brick_heights[current_brick]
                        ax[idx_individual].bar(idx_col, current_height, bottom=column_height, color=cm.jet(current_brick/len(self.brick_heights)))
                        column_height += current_height
            plt.show()
        
        interact(print_results, generation=widgets.IntSlider(min=0,max=len(self.history)-1,step=1,value=1))

    def plot_best_individual(self):
        '''
            Method to plot the best individual.
        '''
        _,_,best_individual,best_fitness = self.history[-1]
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        fig.suptitle(f'Best individual (f = {best_fitness:.3f})')
        for idx_col in range(len(best_individual)):
            column_height = 0
            for idx_bri in range(len(best_individual[idx_col])):
                current_brick = best_individual[idx_col][idx_bri]
                current_height = self.brick_heights[current_brick]
                ax.bar(idx_col, current_height, bottom=column_height, color=cm.jet(current_brick/len(self.brick_heights)))
                column_height += current_height
                ax.text(idx_col, column_height-current_height/2, f"{current_height:5.1f}", ha='center', va='center')
            ax.text(idx_col, column_height, f"{column_height:5.1f}", ha='center', va='bottom')
        plt.show()