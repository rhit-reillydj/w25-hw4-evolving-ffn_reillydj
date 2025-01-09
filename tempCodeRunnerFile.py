        parents = random.choices(population, weights=[ind.fitness() for ind in population], k=2)
