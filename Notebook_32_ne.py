import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from multiprocessing import Process, Array
import seaborn as sns
sns.set()

symbol = 'GOOG'
if not os.path.exists(f'Symbols/{symbol}.csv'):
    import pandas_datareader as pdr
    df = pdr.get_data_yahoo(symbol)
    df.to_csv(f'Symbols/{symbol}.csv')
df = pd.read_csv(f'Symbols/{symbol}.csv')
print(df.head())

close = df.Close.values.tolist()
initial_money = 10000
window_size = 30
skip = 1

"""
close high low open
add movement
movement + google attention on (another stock)
movement + google attention on elmo news

load and use it for other stock
"""


class Portfolio:
    def __init__(self, window_size, trend, skip, initial_money, commission=.01):
        self.window_size = window_size
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.commission = commission

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res]).astype(np.float32)

    def buy(self, individual):
        initial_money = self.initial_money
        starting_money = initial_money
        state = self.get_state(0)
        inventory = []
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action = np.argmax(individual(state, training=False))
            next_state = self.get_state(t + 1)

            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t] * (1 + self.commission)
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, self.trend[t], initial_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t] * (1 - self.commission)
                states_sell.append(t)
                try:
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.trend[t], invest, initial_money)
                )
            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def call(self, genome, i, fitnesses):
        initial_money = self.initial_money
        starting_money = initial_money
        state = self.get_state(0)
        inventory = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action = np.argmax(genome(state, training=False))
            next_state = self.get_state(t + 1)

            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t] * (1 + self.commission)

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t] * (1 - self.commission)

            state = next_state
        invest = ((starting_money - initial_money) / initial_money) * 100
        fitnesses[i] = invest


def net(name, hidden_size=128):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(window_size, ), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(-window_size ** -.5, window_size ** -.5)),
        tf.keras.layers.LeakyReLU(),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3, use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(-hidden_size ** -.5, hidden_size ** -.5)),
        tf.keras.layers.Softmax(),
    ], name=str(name))


class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, time_machine):
        self.population = [net(i) for i in range(population_size)]
        self.mutation_rate = mutation_rate
        self.time_machine = time_machine

    def mutate(self, individual, scale=1.0):
        for layer in individual.layers:
            weights = [w + np.random.normal(loc=0, scale=scale, size=w.shape) * np.random.binomial(1, p=self.mutation_rate, size=w.shape) for w in layer.get_weights()]
            layer.set_weights(weights)
        return individual

    def crossover(self, parent1, parent2):
        child1 = net((int(parent1.name) + 1) * 10)
        child1.set_weights(parent1.get_weights())
        child2 = net((int(parent1.name) + 1) * 10)
        child2.set_weights(parent2.get_weights())
        for i, J in enumerate(parent1.layers):
            ws1 = []
            ws2 = []
            for j, _ in enumerate(J.get_weights()):
                ws1.append(child1.layers[i].get_weights()[j])
                ws2.append(child2.layers[i].get_weights()[j])
                if len(ws1[-1].shape) != 2:
                    continue
                n_neurons = ws1[-1].shape[1]
                cutoff = np.random.randint(0, n_neurons)
                ws1[-1][:, cutoff:] = parent2.layers[i].get_weights()[j][:, cutoff:].copy()
                ws2[-1][:, cutoff:] = parent1.layers[i].get_weights()[j][:, cutoff:].copy()
            child1.layers[i].set_weights(ws1)
            child2.layers[i].set_weights(ws2)
        return child1, child2

    def calculate_fitness(self):
        fitnesses = Array('d', [.0] * len(self.population))
        pool = [Process(target=self.time_machine.call, args=(genome, i, fitnesses)) for i, genome in enumerate(self.population)]
        for p in pool:
            p.start()
        for p in pool:
            p.join()
        for i, genome in enumerate(self.population):
            genome.fitness = fitnesses[i]
            print(genome.fitness)

    def evolve(self, generations=20, checkpoint=1):
        n_winners = int(len(self.population) * 0.4)
        n_parents = len(self.population) - n_winners
        for epoch in range(generations):
            self.calculate_fitness()
            fitnesses = [i.fitness for i in self.population]
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sort_fitness]
            fittest_individual = self.population[0]
            if (epoch + 1) % checkpoint == 0:
                print('epoch %d, fittest individual %d with accuracy %f' % (epoch + 1, sort_fitness[0],
                                                                            fittest_individual.fitness))
            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
            parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)
            for i in np.arange(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                next_population += [self.mutate(child1), self.mutate(child2)]
            self.population = next_population
        return fittest_individual


portfolio = Portfolio(window_size, close, skip, initial_money, commission=.0)

population_size = 20
generations = 4
mutation_rate = 0.1
neural_evolve = NeuroEvolution(population_size, mutation_rate, portfolio)

fittest_nets = neural_evolve.evolve(generations)
states_buy, states_sell, total_gains, invest = portfolio.buy(fittest_nets)


fig = plt.figure(figsize=(15, 5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
plt.legend()
plt.show()
