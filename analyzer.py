import pandas as pd
import numpy as np
from itertools import tee, izip
import random
import matplotlib.pyplot as plt

CHROMOSOME_SIZE = 5
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.2
GENERATIONS = 100
POPULATION_SIZE = 20
NO_FEATURES = 9

def get_music_components(activity, df, label):
    activity = activity.split(",")
    mc = set()
    for a in activity:
        df2= (df.loc[a][df.columns.difference(['Activity'])] == label)
        df2 = df2.reset_index()
        mc.update((df2[df2[a] == True])['index'])
    return mc

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def get_music_distance(chromosome):
    fdist = 0
    for (i1, row1), (i2, row2) in pairwise(chromosome.iterrows()):
        dist = 0
        dist = dist + np.linalg.norm(row1['acousticness']-row2['acousticness'])
        dist = dist + np.linalg.norm(row1['tempo']-row2['tempo'])
        dist = dist + np.linalg.norm(row1['instrumentalness']-row2['instrumentalness'])
        dist = dist + np.linalg.norm(row1['speechiness']-row2['speechiness'])
        dist = dist + np.linalg.norm(row1['loudness']-row2['loudness'])
        dist = dist + np.linalg.norm(row1['valence']-row2['valence'])
        dist = dist + np.linalg.norm(row1['liveness']-row2['liveness'])
        dist = dist + np.linalg.norm(row1['danceability']-row2['danceability'])
        dist = dist + np.linalg.norm(row1['energy']-row2['energy'])
        fdist = max(fdist, dist)
        #print dist
    #print fdist
    fdist = fdist/(NO_FEATURES)
    return fdist


def fitness_function(pmc, nmc, chromosome):
    total_score = 0
    for m in pmc:
        total_score= total_score+chromosome[m].sum()

    for m in nmc:
        total_score= total_score-chromosome[m].sum()
    total_score = total_score/((len(nmc) + len(pmc))*CHROMOSOME_SIZE)
    dist = get_music_distance(chromosome)

    fin_score = total_score*0.5-dist*0.5

    #print fin_score
    return fin_score

def sigma_scaling(score):
    npscore = np.array(score)
    if(np.std(npscore) != 0 ):
        sigma_score = 1 + (npscore - np.mean(npscore))/(2*np.std(npscore))
        sigma_score = [0 if i < 0 else i for i in sigma_score]
        return sigma_score
    return score

def normalize_score(score):
    npscore = np.array(score)
    scale_score = (npscore - np.min(npscore))/(np.max(npscore) - np.min(npscore))
    return scale_score

def get_parent_chromosome_roulette(population, score, cumulative):
    #perform roulette wheel selection method
    r = random.uniform(0, (cumulative))
    temp_cl = 0;
    for i in range(0, POPULATION_SIZE):
        if(temp_cl > r):
            return population[i]
        temp_cl = temp_cl + score[i]
    return population[-1]

def crossover_parents(chromo_one, chromo_two):
    r = random.uniform(0, 1.0)
    if(r < CROSSOVER_RATE):
        p = random.randint(0, CHROMOSOME_SIZE-1)
        b, c = chromo_one.head(p).copy(), chromo_two.head(p).copy()
        d, e = chromo_one.tail(CHROMOSOME_SIZE-p).copy(), chromo_two.tail(CHROMOSOME_SIZE-p).copy()

        offspring_one = b.append(e, ignore_index=True)
        offspring_two = c.append(d, ignore_index=True)
        return offspring_one, offspring_two

    else:
        return chromo_one, chromo_two

def mutate_chromosome(chromosome, df):
    r = random.uniform(0, 1.0)
    if(r < MUTATION_RATE):
        p = random.randint(0, CHROMOSOME_SIZE-1)
        mus_gene = df.sample(n=1)
        mut_chrom = chromosome.copy()
        mut_chrom.iloc[p] = mus_gene.iloc[0]
        return mut_chrom
    else:
        return chromosome

def get_fitness_score(pmc, nmc, initial_population):
    score = []
    for ip in range(len(initial_population)):
        score = score[:] + [fitness_function(pmc, nmc, initial_population[ip])]
        #score.append()
    #score = normalize_score(score)
    return score, sum(score)


def get_music_rec(pmc, nmc, df):
    #1. Initalize the chromose population randomly
    initial_population = []
    for i in range(0, POPULATION_SIZE):
        initial_population.append(df.sample(n=CHROMOSOME_SIZE))
    #print initial_population

    score_keeper = []
    #2. Produce the next population through crossover and mutation
    # NATURAL SELECTION: Roulette wheel works best for recommender systems
    for j in range(GENERATIONS):
        score, cumulative = get_fitness_score(pmc, nmc, initial_population)
        new_population = []
        score_keeper.append(sum(score)/POPULATION_SIZE)
        score = list(score)
        print score
        print max(score)
        new_population.append(initial_population[score.index(max(score))])
        while len(new_population) != POPULATION_SIZE:
            #(i) Select two parent chromosomes for reproduction
            p1 =  get_parent_chromosome_roulette(initial_population, score, cumulative)
            p2 =  get_parent_chromosome_roulette(initial_population, score, cumulative)
            #print p1['song_title'], p2['song_title']

            #(ii) Crossover with a probability of CROSSOVER_RATE
            c1, c2 = crossover_parents(p1, p2)
            #print c1['song_title'], c2['song_title']

            #(iii) Mutation with a probability of MUTATION_RATE
            c1 = mutate_chromosome(c1, df)
            c2 = mutate_chromosome(c2, df)

            #s, cumulative = get_fitness_score(pmc, nmc, new_population)
            #print "Gen: ",  j, s


            new_population.append(c1)


        initial_population = new_population[:]

    plt.title("Genetic Algorithm Run")
    plt.xlabel("Generations")
    plt.ylabel("Fitness function")
    plt.plot(range(GENERATIONS), score_keeper)
    plt.show()
    final_score, cl = get_fitness_score(pmc, nmc, initial_population)
    print max(final_score)
    optim_chromo = initial_population[final_score.index(max(final_score))]
    optim_chromo.to_csv("./data/Concentration.csv", sep=',')
    return optim_chromo



if __name__ == '__main__':
    import sys
    activity = sys.argv[1]
    df = pd.read_csv("./data/pdata.csv")
    d2 = pd.read_csv("./data/det.csv")
    d2.set_index('Activity', inplace=True)
    #print df.describe()
    #print d2.head()
    pmc = get_music_components(activity, d2, 'H')

    nmc = get_music_components(activity, d2, 'L')

    print pmc, nmc
    print get_music_rec(pmc, nmc, df)
