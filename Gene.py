import numpy as np
import random

class Chromosome:
    def __init__(self, weights ):
        self.weights = weights
        self.fitness = []

    def set_fitness( self, fitness ):
        self.fitness.append(fitness)
    def get_fitness(self):
        return self.fitness
    def get_weights(self):
        return self.weights

class Population:
    def __init__(self, p_fittest, p_mutation, size, shapes):
        
        self.size = size
        self.shepas = shapes
        
        self.p_fittest = p_fittest
        self.p_mutation = p_mutation
        self.chromosomes = []
        
        for _ in range(size):
            weights = []
            for s in shapes:
                k = np.random.uniform(-100, 100, s)
                weights.append(k)
            self.chromosomes.append( Chromosome(weights) )
    
    def get_chromosomes(self):
        return self.chromosomes

    def get_best_(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda x: np.mean(x.get_fitness()), reverse=True)
        return self.chromosomes[0]

    def selection(self):
        # fitness = [ np.mean(x.get_fitness()) for x in self.chromosomes]

        # total_fit = float(sum(fitness))
        # relative_fitness = [f/total_fit for f in fitness]
        # probabilities = [sum(relative_fitness[:i+1]) 
        #              for i in range(len(relative_fitness))]

        p_fittest = int(self.size * self.p_fittest)
        if p_fittest % 2 == 1:
            p_fittest -= 1
        offspring_size = self.size - p_fittest
        fittest = self.chromosomes[:p_fittest]
        
        # selection = self.roulette_wheel_pop(probabilities, offspring_size)
        # shape = (int(len(selection)/2),2)
        # selection = np.array(selection).reshape(shape) 
        

        selection = []
        luckily = self.chromosomes

        offspring_size = int(self.size/2) - int(len(fittest)/2)
        for _ in range(offspring_size):
            selection.append([random.choice(fittest),random.choice(luckily)])
        
        

        return fittest, selection

    def roulette_wheel_pop(self, probabilities, number):
        chosen = []
        for _ in range(number):
            r = random.random()
            for (i, individual) in enumerate(self.chromosomes):
                if r <= probabilities[i]:
                    chosen.append(individual)
                    break
        return chosen 

    def new_generation(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda x: np.mean(x.get_fitness()), reverse=True)
        self.chromosomes = self.crossover()
    
    def crossover( self):
        fittest, selection = self.selection()
        n_generation = []
        for selected in selection:
            fw = selected[0].get_weights()
            mw = selected[1].get_weights()

            s1 = []
            s2 = []

            for i in range(len(fw)):
                f = fw[i].copy().reshape(fw[i].size)
                m = mw[i].copy().reshape(mw[i].size)
                x  = random.randint(1, fw[i].size)
                tmp = f[:x].copy()
                f[:x], m[:x]  = m[:x], tmp
                s1.append(f.reshape(fw[i].shape))
                s2.append(m.reshape(mw[i].shape))
                 
            n_generation.append( Chromosome(s1) )
            n_generation.append( Chromosome(s2) )

        self.mutation(n_generation)
        return fittest + n_generation

    def mutation( self, offsprings ):
        for off in offsprings:
            p = random.random()
            if p <= self.p_mutation:
                offw = off.get_weights()
                n_mutation  = random.randint(1, 10)
                for _ in range(n_mutation):
                    s = random.randint(0, len(offw) - 1)
                    
                    if len(offw[s].shape) > 1:
                        x = random.randint(0, offw[s].shape[0] - 1)
                        y = random.randint(0, offw[s].shape[1] - 1)

                        offw[s][x][y]= random.uniform(-100,100)
                    





