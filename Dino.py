from keyboardEvent import PressKey, ReleaseKey, VK_UP, VK_DOWN
from sklearn.neural_network import MLPClassifier, MLPRegressor
from Gene import Chromosome, Population
from grab_screen import GameScreen
import numpy as np
import os.path
import time

def press(k):
    PressKey(k)
    ReleaseKey(k)

def save( weights, filename ):
    filename = 'DinoModels/' + filename
    np.save(filename, weights)

def load( filename ):
    filename = 'DinoModels/' + filename
    return np.load(filename)

def play(gs, best_player):
    clf = MLPClassifier(hidden_layer_sizes=(5,))
    clf.fit([[0,0,0,0]], [[0,0,0]])
    coefs = clf.coefs_
    shapes = []
    for coef in coefs:
        shapes.append( coef.shape )

    print( shapes )
    p = Population(0.3, 0.2, 20, shapes )

    for _ in range(1000):
        for _, _ in enumerate(p.get_chromosomes()):
            clf.coefs_ = best_player
            total = 0

            PressKey(VK_UP)
            time.sleep(2)
            ReleaseKey(VK_UP)

            lt = time.time()
            pd = 1000
            while True:
                _, _, obs = gs.get_screen_process()
                dif_time = time.time() - lt
                
                if not obs:
                    d, l, a = 1000, 0, 0
                else:
                    d, l, a = obs[0]
                
                if d > pd:
                    total += 1
                else:
                    fps = 1/dif_time
                    v = (pd - d) * fps

                pd = d
                y_pred = clf.predict([[d,a,l,v]])

                if y_pred[0][0] == 1:
                    press(VK_UP)
                if y_pred[0][1] == 1:
                    press(VK_DOWN)
                    
                if gs.is_game_over():
                    break
                
                lt = time.time()

def learning( gs ):
    clf = MLPClassifier(hidden_layer_sizes=(5,))
    clf.fit([[0,0,0,0]], [[0,0,0]])
    coefs = clf.coefs_
    shapes = []
    for coef in coefs:
        shapes.append( coef.shape )

    print( shapes )
    p = Population(0.3, 0.2, 20, shapes )

    for i in range(1000):
        for idx, gene in enumerate(p.get_chromosomes()):
            clf.coefs_ = gene.get_weights()
            total = 0

            PressKey(VK_UP)
            time.sleep(2)
            ReleaseKey(VK_UP)

            lt = time.time()
            pd = 1000
            while True:
                _, _, obs = gs.get_screen_process()
                dif_time = time.time() - lt
                
                if not obs:
                    d, l, a = 1000, 0, 0
                else:
                    d, l, a = obs[0]
                
                if d > pd:
                    total += 1
                else:
                    fps = 1/dif_time
                    v = (pd - d) * fps

                pd = d
                y_pred = clf.predict([[d,a,l,v]])

                if y_pred[0][0] == 1:
                    press(VK_UP)
                if y_pred[0][1] == 1:
                    press(VK_DOWN)
                    
                if gs.is_game_over():
                    break
                
                #print('loop took {}'.format(dif_time))
                #print('Score {}, Distancia {}, altura {}, lagura {}, velocidade {}'.format(total,d,a,l,v))
                lt = time.time()
                
            gene.set_fitness( total )
            print('Gene {} score: {} - {} - {}'.format( idx, total, gene.get_fitness(), np.max(gene.get_fitness())))
            
        
        best = p.get_best_()
        best.get_weights()
        filename = 'score_{}.npy'.format( np.max(best.get_fitness()) )
        save(best.get_weights(), filename)
        print('Best generation {}, is {}'.format(i, best.get_fitness() ))
        p.new_generation()

def main():
    gs = GameScreen()
    if not gs.game_found:
        print('ERROR')
        exit()

    for i in range(5,0,-1):
        time.sleep(1)
        print('Start in {}'.format(i))
    
    learning(gs)
    
    # filename = 'score_72.npy'
    # play(gs, load(filename))

if __name__ == '__main__':
    main()