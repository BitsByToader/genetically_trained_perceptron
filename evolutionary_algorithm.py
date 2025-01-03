import random
import math
from ioptimization_problem import IOptimizationProblem


class Chromosome:
    def __init__(self, no_genes, min_values, max_values):
        self.no_genes = no_genes
        self.genes = [0.0] * no_genes
        self.min_values = list(min_values)
        self.max_values = list(max_values)
        self.fitness = 0.0
        self._initialize_genes()

    def _initialize_genes(self):
        for i in range(self.no_genes):
            self.genes[i] = self.min_values[i] + random.random() * (self.max_values[i] - self.min_values[i])

    def __copy__(self):
        return Chromosome(self.no_genes, self.min_values, self.max_values)

    def copy_from(self, other):
        self.no_genes = other.no_genes
        self.genes = list(other.genes)
        self.min_values = list(other.min_values)
        self.max_values = list(other.max_values)
        self.fitness = other.fitness


class Selection:
    @staticmethod
    def tournament(population):
        return random.choice(population)

    @staticmethod
    def get_best(population):
        return max(population, key=lambda c: c.fitness)

class Crossover:
    @staticmethod
    def arithmetic(mother, father, rate):
        if random.random() < rate:
            child = Chromosome(mother.no_genes, mother.min_values, mother.max_values)
            idx = random.randint(0, mother.no_genes-1)
            for i in range(0, idx):
                child.genes[i] = mother.genes[i]
            for i in range(idx, child.no_genes):
                child.genes[i] = father.genes[i]

            return child
        else:
            return father

class Mutation:
    @staticmethod
    def reset(child, rate):
        if random.random() < rate:
            gene_idx = random.randint(0, child.no_genes-1)
            child.genes[gene_idx] = child.min_values[gene_idx] + random.random() * (child.max_values[gene_idx] - child.min_values[gene_idx])


class EvolutionaryAlgorithm:
    def solve(self, problem, population_size, max_generations, crossover_rate, mutation_rate):

        population = [problem.make_chromosome() for _ in range(population_size)]
        for individual in population:
            problem.compute_fitness(individual)

        for gen in range(max_generations):
            new_population = [Selection.get_best(population)]

            for i in range(1, population_size):
                # Select parents for crossing.
                p1 = Selection.tournament(population)
                p2 = Selection.tournament(population)

                # Generate child from parents
                c = Crossover.arithmetic(p1, p2, crossover_rate)

                # Mutate child
                Mutation.reset(c, mutation_rate)

                # Compute fitness of new child.
                problem.compute_fitness(c)

                # Add child to next generation population
                new_population.append(c)

            population = new_population

        return Selection.get_best(population)