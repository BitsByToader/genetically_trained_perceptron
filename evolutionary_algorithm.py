import random
import math
from ioptimization_problem import IOptimizationProblem
from chromosome import Chromosome


class Selection:
    @staticmethod
    def tournament(population: [Chromosome], k: int) -> Chromosome:
        members = []
        for i in range(0,k):
            members.append(random.choice(population))
        return max(members, key=lambda c: c.fitness)

    @staticmethod
    def get_best(population: [Chromosome]) -> Chromosome:
        return max(population, key=lambda c: c.fitness)

class Crossover:
    @staticmethod
    def point(mother: Chromosome, father: Chromosome, rate: float) -> Chromosome:
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

    def arithmetic(mother: Chromosome, father: Chromosome, rate: float) -> Chromosome:
        if random.random() < rate:
            child = Chromosome(mother.no_genes, mother.min_values, mother.max_values)
            
            a = random.random()
            for i in range(child.no_genes):
                child.genes[i] = a * mother.genes[i] + (1.0-a) * father.genes[i]

            return child
        else:
            return father

class Mutation:
    @staticmethod
    def reset(child: Chromosome, rate: float):
        if random.random() < rate:
            gene_idx = random.randint(0, child.no_genes-1)
            child.genes[gene_idx] = child.min_values[gene_idx] + random.random() * (child.max_values[gene_idx] - child.min_values[gene_idx])


class EvolutionaryAlgorithm:
    def solve(self, problem: IOptimizationProblem, population_size: int, max_generations: int, crossover_rate: float, mutation_rate: float) -> Chromosome:
        # Create random population of given size
        population = [problem.make_chromosome() for _ in range(population_size)]
        
        # Initialize population for algorithm
        for individual in population:
            problem.compute_fitness(individual)

        for gen in range(max_generations):
            print(f'Begin generation {gen}')
            new_population = [Selection.get_best(population)]

            mean_fitness: float = 0.0

            for i in range(1, population_size):
                # Select parents for crossing.
                p1 = Selection.tournament(population, 2)
                p2 = Selection.tournament(population, 2)

                # Generate child from parents
                c = Crossover.arithmetic(p1, p2, crossover_rate)

                # Mutate child
                Mutation.reset(c, mutation_rate)

                # Compute fitness of new child.
                problem.compute_fitness(c)
                mean_fitness += c.fitness

                # Add child to next generation population
                new_population.append(c)

            population = new_population

            # TODO: Add mean fitness to a report and return at the end of the training.
            # Optionally remove printing.
            mean_fitness /= population_size
            print(f'Generation {gen} had mean fitness of: {mean_fitness}')

        return Selection.get_best(population)