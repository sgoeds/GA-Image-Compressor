# Copyright Samuel Goedert, all rights reserved.

import numpy as np

class GeneticImagePopulation:
    
    def __init__(self, target_img, pop_size, mutation_chance, rectangle_count, heuristic="DEFAULT"):
        if pop_size % 6 != 0:
            # the population size must be divisible by 2 for mating and 3 for elimination, hence 6
            raise Exception("Population size must be divisible by 6")
        self.X, self.Y, _ = target_img.shape
        self.target = target_img
        self.pop = np.random.random((pop_size, rectangle_count, 8))
        self.pop_size = pop_size
        self.mutation_chance = mutation_chance
        self.rectangle_count = rectangle_count
        self.heuristic = self.default_heuristic if heuristic=="DEFAULT" else heuristic
        
    def sort(self):
        heuristics = np.array(list(map(self.heuristic, self.pop)))
#         heuristics = np.array([self.heuristic(i) for i in self.pop])
        indices = np.argsort(heuristics)
        self.pop = self.pop[indices]
        return heuristics[indices[0]]
    
#     def heuristic_vec(self, pop): # deprecated
#         difs = np.array([self.make_image(entity) - self.target for entity in pop])
#         return np.sum(np.abs(difs), axis=(1,2,3))
    
    def naturally_select(self):
        LIVE_COUNT = self.pop_size // 3
        survivors = self.pop[:LIVE_COUNT]
        shape = (self.pop_size - LIVE_COUNT, self.rectangle_count, 8)
        moms = np.zeros(shape)
        dads = np.zeros(shape)
        
        for i in range(4):
            np.random.shuffle(survivors)
            pos = (i*LIVE_COUNT)//2
            pos2 = pos + LIVE_COUNT//2
            moms[pos:pos2] = survivors[:LIVE_COUNT//2]
            dads[pos:pos2] = survivors[LIVE_COUNT//2:]
        self.pop[LIVE_COUNT:] = self.mate(moms, dads)
        
    def mate(self, moms, dads):
        decider = np.random.random((moms.shape[0:2])) > 0.5
        children = np.zeros(moms.shape)
        inverse_decider = np.invert(decider)
        children[decider] = moms[decider]
        children[inverse_decider] = dads[inverse_decider]
        return self.mutate(children)
    
    def mutate(self, e):
        replacement_indices = np.random.random(e.shape) <= self.mutation_chance
        e[replacement_indices] = np.random.random(e.shape)[replacement_indices]
        return e
    
    def default_heuristic(self, entity):
        return np.sum(np.abs(self.make_image(entity) - self.target))
    
    def make_image(self, entity):
        h_img = np.zeros(self.target.shape, dtype=np.int)
        for rec in entity:
            a = rec[0]
            b = rec[1]
            x1 = int(self.X * (a if a < b else b))
            x2 = int(self.X * (b if a < b else a))
            a = rec[2]
            b = rec[3]
            y1 = int(self.Y * (a if a < b else b))
            y2 = int(self.Y * (b if a < b else a))

            a = rec[7]
            chans = rec[4:7]
            h_img[x1: x2, y1: y2] = 255*(chans * a / 1 + (h_img[x1: x2, y1: y2]/255) * (1 - a))
        return h_img
    
    
    

class LayeredGeneticImage:
    
    def __init__(self, target_img, population_sizes, mutation_chances, rectangle_counts):
        self.target = target_img
        self.layer_count = len(population_sizes)
        self.X, self.Y, _ = target_img.shape
        self.pops = []
        self.layer = 0
        self.set_layer(0)
        for i in range(len(population_sizes)):
            self.pops.append(
            GeneticImagePopulation(target_img,
                                  population_sizes[i],
                                  mutation_chances[i],
                                  rectangle_counts[i],
                                  heuristic=self.heuristic))
        
    def heuristic(self, entity):
        return np.sum(np.abs(self.make_image(entity) - self.target))
    
    def set_layer(self, number):
        if number < self.layer:
            self.layer = 0
        if self.layer == 0:
            self.bg = np.zeros(self.target.shape, dtype=np.int)
        for i in range(self.layer, number):
            self.bg = self.make_image(self.pops[i].pop[0])
        self.layer = number
        
    def make_image(self, entity):
        # only difference between this and the one in GeneticImagePopulation is the copy of the background instead of creating a blank image
        h_img = self.bg.copy()
        for rec in entity:
            a = rec[0]
            b = rec[1]
            x1 = int(self.X * (a if a < b else b))
            x2 = int(self.X * (b if a < b else a))
            a = rec[2]
            b = rec[3]
            y1 = int(self.Y * (a if a < b else b))
            y2 = int(self.Y * (b if a < b else a))

            a = rec[7]
            chans = rec[4:7]
            h_img[x1: x2, y1: y2] = 255*(chans * a / 1 + (h_img[x1: x2, y1: y2]/255) * (1 - a))
        return h_img
    
    def final_image(self):
        self.set_layer(0)
        return self.make_image(self.final())
    
    def final(self):
        return np.concatenate([self.pops[i].pop[0] for i in range(self.layer_count)], axis=0)