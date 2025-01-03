from chromosome import Chromosome


# Interface for optimization problems
class IOptimizationProblem:
    # Computes the fitness of the given chromosome according to this optimization problem.
    def compute_fitness(self, chromosome: Chromosome):
        raise NotImplementedError("This method needs to be implemented by a subclass")

    # Creates a new chromosome specific to this optimization problem.
    def make_chromosome(self) -> Chromosome:
        raise NotImplementedError("This method needs to be implemented by a subclass")