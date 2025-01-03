import random
import math

# Interface for optimization problems
class IOptimizationProblem:
    def compute_fitness(self, chromosome):
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def make_chromosome(self):
        raise NotImplementedError("This method needs to be implemented by a subclass")


class Equation(IOptimizationProblem):
    def make_chromosome(self):
        # un cromozom are o gena (x) care poate lua valori in intervalul (-5, 5)
        return Chromosome(1, [-5], [5])

    def compute_fitness(self, chromosome):
        gene = chromosome.genes[0]
        chromosome.fitness = -(abs(gene**5-5*gene+5))


class Fence(IOptimizationProblem):
    def make_chromosome(self):
        # un cromozom are doua gene (x si y) care pot lua valori in intervalul (0, 100)
        return Chromosome(2, [0, 0], [100, 100])

    def compute_fitness(self, chromosome):
        x = chromosome.genes[0]
        y = chromosome.genes[1]
        
        if 2*x + y > 100.0:
            r = 100.0 / (2*x + y)
            x = x * r
            y = y * r

        chromosome.fitness = x * y


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
                # selectare 2 parinti: Selection.tournament
                p1 = Selection.tournament(population)
                p2 = Selection.tournament(population)

                # generarea unui copil prin aplicare crossover: Crossover.arithmetic
                c = Crossover.arithmetic(p1, p2, crossover_rate)

                # aplicare mutatie asupra copilului: Mutation.reset
                Mutation.reset(c, mutation_rate)

                # calculare fitness pentru copil: compute_fitness din problema p
                problem.compute_fitness(c)

                # introducere copil in new_population
                new_population.append(c)

            population = new_population

        return Selection.get_best(population)


if __name__ == "__main__":
    ea = EvolutionaryAlgorithm()

    solution = ea.solve(Equation(), 50, 1000, 0.9, 0.1)  # de completat parametrii algoritmului
    # se foloseste -solution.Fitness pentru ca algoritmul evolutiv maximizeaza, iar aici avem o problema de minimizare
    print(f"{solution.genes[0]:.6f} -> {-solution.fitness:.6f}")
          
    solution = ea.solve(Fence(), 50, 1000, 0.9, 0.1)  # de completat parametrii algoritmului
    print(f"{solution.genes[0]:.2f} {solution.genes[1]:.2f} -> {solution.fitness:.4f}")
