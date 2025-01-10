import math
import random

NUM_INPUT_NEURONS = 2
NUM_HIDDEN_NEURONS = 15     # As low as 7 works
NUM_OUTPUT_NEURONS = 7

POPULATION_SIZE = 50
MAX_GENERATIONS = 10000

MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.1
    
def RELU(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class FFN:
    def __init__(self):
        # Initialize 3D Array for all weights in the network 
        # Index 0: 0 or 1 indicated layer of connections (input-hidden or hidden-output)
        # Index 1: Indexing the neuron that the connection is coming out of
        # Index 2: Indexing the neuron that the connection is running in to
        self.weights = [[[(random.random() * 2) - 1 for _ in range(NUM_HIDDEN_NEURONS)] for _ in range(NUM_INPUT_NEURONS)],     # 2D Array describing connections
                        [[(random.random() * 2) - 1 for _ in range(NUM_OUTPUT_NEURONS)] for _ in range(NUM_HIDDEN_NEURONS)]]    # 2D Array describing connections
        
        # 2D Array for biases
        # Index 0: 0 or 1 indicating the hidden or output layer
        self.biases = [[(random.random() * 2) - 1 for _ in range(NUM_HIDDEN_NEURONS)],
                       [(random.random() * 2) - 1 for _ in range(NUM_OUTPUT_NEURONS)]]
        
        # 2D Array describing activation from each neuron
        # 0: hidden, 1: output
        self.activations = [[0 for _ in range(NUM_HIDDEN_NEURONS)],
                            [0 for _ in range(NUM_OUTPUT_NEURONS)]]
        
    def forwardPass(self, input):
        # Connection Layer 1
        for hidden_index in range(NUM_HIDDEN_NEURONS):
            hidden_sum = 0
            for input_index in range(NUM_INPUT_NEURONS):
                hidden_sum += input[input_index] * self.weights[0][input_index][hidden_index]       # Input value * weight for the connection
            
            hidden_sum += self.biases[0][hidden_index]
            self.activations[0][hidden_index] = RELU(hidden_sum)        # RELU on hidden layer
                
        # Connections Layer 2
        for output_index in range(NUM_OUTPUT_NEURONS):
            output_sum = 0
            for hidden_index in range(NUM_HIDDEN_NEURONS):
                output_sum += self.activations[0][hidden_index] * self.weights[1][hidden_index][output_index]       # Input value * weight for the connection
                
            output_sum += self.biases[1][output_index]
            self.activations[1][output_index] = sigmoid(output_sum)     # Sigmoid on output layer
        
        # Returns list size 3: NOT, OR, AND
        return self.activations[1]      # Output Layer Activations
    

    def fitness(self):
        # Define the truth table inputs and targets
        inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        
        # Target outputs for each gate
        targets_NOT = [1, 1, 0, 0]  # NOT x1
        targets_OR  = [0, 1, 1, 1]  # x1 OR x2
        targets_AND = [0, 0, 0, 1]  # x1 AND x2
        targets_NAND = [1, 1, 1, 0]  # x1 NAND x2
        targets_NOR = [1, 0, 0, 0]  # x1 NOR x2
        targets_XOR = [0, 1, 1, 0]  # x1 XOR x2
        targets_XNOR = [1, 0, 0, 1]  # x1 XNOR x2


        # Initialize MSE sums
        mse_NOT = 0
        mse_OR = 0
        mse_AND = 0
        mse_NAND = 0
        mse_NOR = 0
        mse_XOR = 0
        mse_XNOR = 0

        # Calculate predictions and MSE for each input
        for i, input_pair in enumerate(inputs):
            predictions = self.forwardPass(input_pair)  # Predicted [NOT, OR, AND]

            # Squared errors for each gate
            mse_NOT += (predictions[0] - targets_NOT[i]) ** 2
            mse_OR  += (predictions[1] - targets_OR[i]) ** 2
            mse_AND += (predictions[2] - targets_AND[i]) ** 2
            mse_NAND += (predictions[3] - targets_NAND[i]) ** 2
            mse_NOR  += (predictions[4] - targets_NOR[i]) ** 2
            mse_XOR += (predictions[5] - targets_XOR[i]) ** 2
            mse_XNOR += (predictions[6] - targets_XNOR[i]) ** 2

        # Average over all 4 input cases
        mse_NOT /= 4
        mse_OR /= 4
        mse_AND /= 4
        mse_NAND /= 4
        mse_NOR /= 4
        mse_XOR /= 4
        mse_XNOR /= 4

        # Combine MSE for all gates
        total_mse = mse_NOT + mse_OR + mse_AND + mse_NAND + mse_NOR + mse_XOR + mse_XNOR

        return 7 - total_mse
        
    @classmethod
    def crossover(self, parent1, parent2):
        child = FFN()
        usingParent1 = True
        
        # First layer of weights
        for input_index in range(NUM_INPUT_NEURONS):
            for hidden_index in range(NUM_HIDDEN_NEURONS):
                
                # Take from parent1 or parent2
                if usingParent1:
                    child.weights[0][input_index][hidden_index] = parent1.weights[0][input_index][hidden_index]
                else:
                    child.weights[0][input_index][hidden_index] = parent2.weights[0][input_index][hidden_index]
                    
                # Change which parent we are using
                if random.random() < CROSSOVER_RATE:
                    usingParent1 = not usingParent1
                    
        # Second layer of weights
        for hidden_index in range(NUM_HIDDEN_NEURONS):
            for output_index in range(NUM_OUTPUT_NEURONS):
                
                # Take from parent1 or parent2
                if usingParent1:
                    child.weights[1][hidden_index][output_index] = parent1.weights[1][hidden_index][output_index]
                else:
                    child.weights[1][hidden_index][output_index] = parent2.weights[1][hidden_index][output_index]
                    
                # Change which parent we are using
                if random.random() < CROSSOVER_RATE:
                    usingParent1 = not usingParent1
                    
        # Crossover hidden biases
        for hidden_index in range(NUM_HIDDEN_NEURONS):                
            if usingParent1:
                child.biases[0][hidden_index] = parent1.biases[0][hidden_index]
            else:
                child.biases[0][hidden_index] = parent2.biases[0][hidden_index]
                
            if random.random() < CROSSOVER_RATE:
                usingParent1 = not usingParent1
                
        # Crossover output biases
        for output_index in range(NUM_OUTPUT_NEURONS):
            if usingParent1:
                child.biases[1][output_index] = parent1.biases[1][output_index]
            else:
                child.biases[1][output_index] = parent2.biases[1][output_index]
                
            if random.random() < CROSSOVER_RATE:
                usingParent1 = not usingParent1
                    
        return child
    
    @classmethod
    def copy(self, parent):
        child = FFN()
        
        # First layer of weights
        for input_index in range(NUM_INPUT_NEURONS):
            for hidden_index in range(NUM_HIDDEN_NEURONS):
                child.weights[0][input_index][hidden_index] = parent.weights[0][input_index][hidden_index]
                    
        # Second layer of weights
        for hidden_index in range(NUM_HIDDEN_NEURONS):
            for output_index in range(NUM_OUTPUT_NEURONS):
                child.weights[1][hidden_index][output_index] = parent.weights[1][hidden_index][output_index]
                    
        # Crossover hidden biases
        for hidden_index in range(NUM_HIDDEN_NEURONS):                
            child.biases[0][hidden_index] = parent.biases[0][hidden_index]
                
        # Crossover output biases
        for output_index in range(NUM_OUTPUT_NEURONS):
            child.biases[1][output_index] = parent.biases[1][output_index]
                    
        return child
    
    def checkWorking(self):
        # Define the truth table inputs and targets
        inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        
        # Target outputs for each gate
        targets_NOT = [1, 1, 0, 0]  # NOT x1
        targets_OR  = [0, 1, 1, 1]  # x1 OR x2
        targets_AND = [0, 0, 0, 1]  # x1 AND x2
        targets_NAND = [1, 1, 1, 0]  # x1 NAND x2
        targets_NOR = [1, 0, 0, 0]  # x1 NOR x2
        targets_XOR = [0, 1, 1, 0]  # x1 XOR x2
        targets_XNOR = [1, 0, 0, 1]  # x1 XNOR x2

        # Initialize total
        total = 0
        threshold = 0.5


        # Calculate predictions and MSE for each input
        for i, input_pair in enumerate(inputs):
            predictions = self.forwardPass(input_pair)  # Predicted [NOT, OR, AND]
            binary_predictions = [1 if p > threshold else 0 for p in predictions]


            # Squared errors for each gate
            total += (binary_predictions[0] == targets_NOT[i])
            total += (binary_predictions[1] == targets_OR[i])
            total += (binary_predictions[2] == targets_AND[i])
            total += (binary_predictions[3] == targets_NAND[i])
            total += (binary_predictions[4] == targets_NOR[i])
            total += (binary_predictions[5] == targets_XOR[i])
            total += (binary_predictions[6] == targets_XNOR[i])

        return total == 28
        
    def roundedResults(self):
        print(f'''
        NOT
        [0,0]: {round(self.forwardPass([0,0])[0])}
        [0,1]: {round(self.forwardPass([0,1])[0])}
        [1,0]: {round(self.forwardPass([1,0])[0])}        
        [1,1]: {round(self.forwardPass([1,1])[0])}\n
        OR
        [0,0]: {round(self.forwardPass([0,0])[1])}
        [0,1]: {round(self.forwardPass([0,1])[1])}
        [1,0]: {round(self.forwardPass([1,0])[1])}        
        [1,1]: {round(self.forwardPass([1,1])[1])}\n
        AND
        [0,0]: {round(self.forwardPass([0,0])[2])}
        [0,1]: {round(self.forwardPass([0,1])[2])}
        [1,0]: {round(self.forwardPass([1,0])[2])}        
        [1,1]: {round(self.forwardPass([1,1])[2])}\n
        NAND
        [0,0]: {round(self.forwardPass([0,0])[3])}
        [0,1]: {round(self.forwardPass([0,1])[3])}
        [1,0]: {round(self.forwardPass([1,0])[3])}        
        [1,1]: {round(self.forwardPass([1,1])[3])}\n
        NOR
        [0,0]: {round(self.forwardPass([0,0])[4])}
        [0,1]: {round(self.forwardPass([0,1])[4])}
        [1,0]: {round(self.forwardPass([1,0])[4])}        
        [1,1]: {round(self.forwardPass([1,1])[4])}\n
        XOR
        [0,0]: {round(self.forwardPass([0,0])[5])}
        [0,1]: {round(self.forwardPass([0,1])[5])}
        [1,0]: {round(self.forwardPass([1,0])[5])}        
        [1,1]: {round(self.forwardPass([1,1])[5])}\n
        XNOR
        [0,0]: {round(self.forwardPass([0,0])[6])}
        [0,1]: {round(self.forwardPass([0,1])[6])}
        [1,0]: {round(self.forwardPass([1,0])[6])}        
        [1,1]: {round(self.forwardPass([1,1])[6])}\n''')
    

# ChatGPT Generated Function
def mutate_value(value, mu=0, sigma=0.3, lower_bound=-1, upper_bound=1):
    dynamic_sigma = sigma
    mutated = value + random.gauss(mu, dynamic_sigma)
    return max(lower_bound, min(mutated, upper_bound))
        
        
def mutation(individual):
    
    # First layer of weights
    for input_index in range(NUM_INPUT_NEURONS):
        for hidden_index in range(NUM_HIDDEN_NEURONS):
            if random.random() < MUTATION_RATE:
                individual.weights[0][input_index][hidden_index] = mutate_value(individual.weights[0][input_index][hidden_index])
                
    # Second layer of weights
    for hidden_index in range(NUM_HIDDEN_NEURONS):
        for output_index in range(NUM_OUTPUT_NEURONS):
            if random.random() < MUTATION_RATE:
                individual.weights[1][hidden_index][output_index] = mutate_value(individual.weights[1][hidden_index][output_index])
                
    # Crossover hidden biases
    for hidden_index in range(NUM_HIDDEN_NEURONS):                
        if random.random() < MUTATION_RATE:
            individual.biases[0][hidden_index] = mutate_value(individual.biases[0][hidden_index])            
            
    # Crossover output biases
    for output_index in range(NUM_OUTPUT_NEURONS):
        if random.random() < MUTATION_RATE:
            individual.biases[1][output_index] = mutate_value(individual.biases[1][output_index])
    
    return individual


def nextGeneration(population):
    fittest = population[0]
    highestFitness = 0
    scaled_fitness = []
    
    for individual in population:
        fitness = individual.fitness()
        scaled_fitness.append(fitness ** 2)
        if fitness > highestFitness:
            highestFitness = fitness
            fittest = individual
            
    outputPopulation = [fittest]
    
    for _ in range(POPULATION_SIZE - 1):
        parents = random.choices(population, weights=scaled_fitness, k=2)
        
        offspring = FFN.crossover(parents[0], parents[1])        
        offspring = mutation(offspring)
        
        outputPopulation.append(offspring)
    
    return outputPopulation
        

def main():
    population = [FFN() for _ in range(POPULATION_SIZE)]
    best_individual = population[0]
    generation = 0

    while not best_individual.checkWorking() and generation < MAX_GENERATIONS:
        population = nextGeneration(population)
        best_individual = max(population, key=lambda individual: individual.fitness())


        if generation % 50 == 0:
            print(F"Generation {generation}: Highest fitness {best_individual.fitness()}")
            
        generation += 1
        
    best_individual = max(population, key=lambda individual: individual.fitness())

    if best_individual.checkWorking():
        print(f"Working solution found in generation {generation}")
        best_individual.roundedResults()

    else:
        print("Solution not found")
        best_individual.roundedResults()
            
main()