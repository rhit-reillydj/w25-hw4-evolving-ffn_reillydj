import random
import struct
import numpy as np
from multiprocessing import Pool

NUM_INPUT_NEURONS = 784     # Pixels in image
NUM_HIDDEN_NEURONS = 128
NUM_OUTPUT_NEURONS = 10     # Numbers 0 - 9

POPULATION_SIZE = 100
MAX_GENERATIONS = 500

MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.2

BATCH_SIZE = 10  # Use a subset of the training data for faster evaluation

def load_mnist_images(file_path, num_images_to_read):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images file."
        images = np.frombuffer(f.read(num_images_to_read * rows * cols), dtype=np.uint8)
        images = images.reshape(num_images_to_read, rows * cols).astype(np.float32)
        images /= 255.0
    return images

def load_mnist_labels(file_path, num_labels_to_read):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels file."
        labels = np.frombuffer(f.read(num_labels_to_read), dtype=np.uint8)
        labels = labels.tolist()  # Convert to Python list
    return labels

TRAINING_IMAGES = load_mnist_images('TrainingData/train-images-idx3-ubyte', 10000)
TRAINING_LABELS = load_mnist_labels('TrainingData/train-labels-idx1-ubyte', 10000)

TESTING_IMAGES = load_mnist_images('TrainingData/t10k-images-idx3-ubyte', 100)
TESTING_LABELS = load_mnist_labels('TrainingData/t10k-labels-idx1-ubyte', 100)

def RELU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class FFN:
    def __init__(self):
        # Numpy for array handling is faster
        # 2 Layers of weights between neurons
        self.weights = [
            np.random.uniform(-1, 1, size=(NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS)),
            np.random.uniform(-1, 1, size=(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS))
        ]
        
        # Bias arrays for hidden and output layers
        self.biases = [
            np.random.uniform(-1, 1, size=(NUM_HIDDEN_NEURONS,)),
            np.random.uniform(-1, 1, size=(NUM_OUTPUT_NEURONS,))
        ]
        
        
    def forwardPass(self, input_vector):
        # Calculate hidden layer activations with vectorization
        hidden_sum = np.dot(input_vector, self.weights[0]) + self.biases[0]
        hidden_act = RELU(hidden_sum)

        # Calculate output layer activations with vectorization
        output_sum = np.dot(hidden_act, self.weights[1]) + self.biases[1]
        output_act = sigmoid(output_sum)

        return output_act
    
    
    def fitness(self):
        # Random subset of indices
        batch_indices = random.sample(range(len(TRAINING_IMAGES)), BATCH_SIZE)
        
        # Extract images and labels for the mini-batch
        images_batch = TRAINING_IMAGES[batch_indices]
        labels_batch = np.array(TRAINING_LABELS)[batch_indices]

        # Entire forward pass using numpy vectorization
        hidden_sum = np.dot(images_batch, self.weights[0]) + self.biases[0]
        hidden_act = RELU(hidden_sum)
        output_sum = np.dot(hidden_act, self.weights[1]) + self.biases[1]
        output_act = sigmoid(output_sum)

        # Find predictions and compare to labels
        predictions = np.argmax(output_act, axis=1)
        correct = np.sum(predictions == labels_batch)
        return correct
        
        
    @classmethod
    def crossover(cls, parent1, parent2):
        child = cls()
        usingParent1 = True

        # Crossover for weights[0] (input-hidden)
        for i in range(NUM_INPUT_NEURONS):
            for j in range(NUM_HIDDEN_NEURONS):
                if usingParent1:
                    child.weights[0][i, j] = parent1.weights[0][i, j]
                else:
                    child.weights[0][i, j] = parent2.weights[0][i, j]
                if random.random() < CROSSOVER_RATE:
                    usingParent1 = not usingParent1

        # Crossover for weights[1] (hidden-output)
        for i in range(NUM_HIDDEN_NEURONS):
            for j in range(NUM_OUTPUT_NEURONS):
                if usingParent1:
                    child.weights[1][i, j] = parent1.weights[1][i, j]
                else:
                    child.weights[1][i, j] = parent2.weights[1][i, j]
                if random.random() < CROSSOVER_RATE:
                    usingParent1 = not usingParent1

        # Crossover biases[0] (hidden layer)
        for j in range(NUM_HIDDEN_NEURONS):
            if usingParent1:
                child.biases[0][j] = parent1.biases[0][j]
            else:
                child.biases[0][j] = parent2.biases[0][j]
            if random.random() < CROSSOVER_RATE:
                usingParent1 = not usingParent1

        # Crossover biases[1] (output layer)
        for j in range(NUM_OUTPUT_NEURONS):
            if usingParent1:
                child.biases[1][j] = parent1.biases[1][j]
            else:
                child.biases[1][j] = parent2.biases[1][j]
            if random.random() < CROSSOVER_RATE:
                usingParent1 = not usingParent1

        return child
    
    
    @classmethod
    def copy(cls, parent):
        # Numpy copy is faster than manual copying
        child = cls()
        child.weights[0] = np.copy(parent.weights[0])
        child.weights[1] = np.copy(parent.weights[1])
        child.biases[0] = np.copy(parent.biases[0])
        child.biases[1] = np.copy(parent.biases[1])
        return child    
    
    def runFullTest(self):
        # Check best FFN against test images
        total = 0
        for i, image in enumerate(TESTING_IMAGES):
            output = self.forwardPass(image)
            predicted_label = np.argmax(output)
            if predicted_label == TESTING_LABELS[i]:
                total += 1
        
        percentage = total / len(TESTING_IMAGES)
        print(f"Full Test Accuracy: {percentage * 100}%")
        return percentage >= 0.9 # True if > 90% accuracy


    def results(self):
        # Check best FFN against training images
        total = 0
        batch_indices = random.sample(range(len(TRAINING_IMAGES)), BATCH_SIZE * 100)

        for i in batch_indices:
            output = self.forwardPass(TRAINING_IMAGES[i])
            predicted_label = np.argmax(output)
            total += predicted_label == TRAINING_LABELS[i]
        return total


def mutate_value(value, generation, mu=0, sigma=0.3, lower_bound=-1, upper_bound=1):
    dynamic_sigma = sigma * (1 - generation / MAX_GENERATIONS)
    mutated = value + random.gauss(mu, dynamic_sigma)
    return max(lower_bound, min(mutated, upper_bound))

def mutation(individual, generation):
    # Mutate weights[0]
    mask_0 = np.random.rand(*individual.weights[0].shape) < MUTATION_RATE
    individual.weights[0][mask_0] = [
        mutate_value(val, generation) for val in individual.weights[0][mask_0]
    ]
    
    # Mutate weights[1]
    mask_1 = np.random.rand(*individual.weights[1].shape) < MUTATION_RATE
    individual.weights[1][mask_1] = [
        mutate_value(val, generation) for val in individual.weights[1][mask_1]
    ]
    
    # Mutate biases[0]
    mask_b0 = np.random.rand(*individual.biases[0].shape) < MUTATION_RATE
    individual.biases[0][mask_b0] = [
        mutate_value(val, generation) for val in individual.biases[0][mask_b0]
    ]
    
    # Mutate biases[1]
    mask_b1 = np.random.rand(*individual.biases[1].shape) < MUTATION_RATE
    individual.biases[1][mask_b1] = [
        mutate_value(val, generation) for val in individual.biases[1][mask_b1]
    ]
    
    return individual


def fitness_wrapper(individual):
    return individual.fitness()

def nextGeneration(population, generation):
    # Evaluate fitness in parallel
    with Pool() as p:
        fit_list = p.map(fitness_wrapper, population)

    # Find fittest
    fittest_idx = np.argmax(fit_list)
    highest_fitness = fit_list[fittest_idx]
    fittest = population[fittest_idx]

    # Find fitness squared for roulette selection
    scaled_fitness = [f ** 2 for f in fit_list]

    outputPopulation = [fittest]

    for _ in range(POPULATION_SIZE - 1):
        parents = random.choices(population, weights=scaled_fitness, k=2)
        offspring = FFN.crossover(parents[0], parents[1])
        offspring = mutation(offspring, generation)
        outputPopulation.append(offspring)

    return outputPopulation, highest_fitness

def main():
    population = [FFN() for _ in range(POPULATION_SIZE)]

    # Evaluate once at the start to pick an initial best
    with Pool() as p:
        fit_list = p.map(fitness_wrapper, population)
    best_individual = population[np.argmax(fit_list)]

    generation = 0
    while generation < MAX_GENERATIONS:
        population, highest_fit = nextGeneration(population, generation)
        
        # Run fitness checks in parallel
        with Pool() as p:
            fit_list = p.map(fitness_wrapper, population)
        best_individual = population[np.argmax(fit_list)]
        
        print(f"\nGeneration {generation}")
        accuracy = best_individual.results()
        print(f"Accuracy: {accuracy / 10}%")
        
        if (accuracy / 10) > (BATCH_SIZE * 10) - 5:        # Accuracy out of 100 compared to 95
            print("\nCANDIDATE FOUND! Running full test! :D")
            if best_individual.runFullTest():
                break
        
        generation += 1

    if best_individual.runFullTest():
        print(f"Working solution found in generation {generation}")
        best_individual.roundedResults()
        
    else:
        print("Solution not found. Closest")
        best_individual.runFullTest()

if __name__ == "__main__":
    main()
