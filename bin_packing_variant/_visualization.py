'''
    General visualization methods for the Genetic Brick Stacking problem.
    self is an instance of the GeneticBrickStacking class.
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ipywidgets as widgets
from ipywidgets import interact    

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