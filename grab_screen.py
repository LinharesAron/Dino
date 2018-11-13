import cv2
import os
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
import time

class GameScreen():

    def __init__(self, thresold=0.9):
        self.thresold = thresold
        self.board_size = (150, 400)
        self.game_found = False
        self.min_dist = 30
        self.__find_board_game()

    def get_screen_shot(self):
        if self.game_found:
            original_image = np.array(ImageGrab.grab(bbox=self.game_board))
        else:
            original_image = np.array(ImageGrab.grab())
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def __merge_obs(self, obs):
        r = []
        for ob in obs: 
            x = min([ x for x,_,_,_ in ob ])
            y = min([ y for _,y,_,_ in ob ])
            w = max([ x+w for x,_,w,_ in ob ]) - x
            h = max([ h for _,_,_,h in ob ])
            r.append((x,y,w,h))
        return r

    def __find_merge_obs(self, cons):
        obs = []
        cons.sort(key=lambda con: con[1])
        for _, x, y, w, h in cons:
            if not obs:
                l = []
                l.append((x,y,w,h))
                obs.append(l)
            else:
                closed = False
                for ob in obs:
                    for ox,oy,_,_ in ob:
                        if abs(int(ox-x)) < self.min_dist and abs(int(oy-y)) < self.min_dist:
                            closed = True
                            ob.append((x,y,w,h))
                            break
                if not closed:
                    l = []
                    l.append((x,y,w,h))
                    obs.append(l)
        return self.__merge_obs( obs )

    def get_distance( self, dino, obs ):
        dx, dy, dw, dh = (dino[1], dino[2], dino[3], dino[4])
        dino_limit_x = dx + dw
        dino_limit_y = self.max_y + dh

        r = []
        screen = []
        for x, y, w, h in obs:
            if x > dino_limit_x and y+h > self.max_y:
              r.append( (x - dino_limit_x, w, abs(int((y+h) - dino_limit_y)) ) )
              screen.append((x,y,w,h))

        r.sort()
        return (dx, dy, dw, dh), r, screen

    def is_game_over(self):
        game_over = cv2.imread(os.path.join('Prefabs', 'game_over.PNG'), 0)
        game_over_inverse = cv2.imread(os.path.join('Prefabs', 'game_over_invert.PNG'), 0)

        screen_shot = self.get_screen_shot()
        res = cv2.matchTemplate(screen_shot, game_over, cv2.TM_CCOEFF_NORMED)
        res_invert = cv2.matchTemplate(screen_shot, game_over_inverse, cv2.TM_CCOEFF_NORMED)

        _, m, _, _ = cv2.minMaxLoc(res)
        _, m_i, _, _ = cv2.minMaxLoc(res_invert)
        return m > self.thresold or m_i > self.thresold

    def get_screen_process(self):
        
        # img = cv2.imread('TESTE.png')
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray  = cv2.bitwise_not(gray)

        img = self.get_screen_shot()
        if img[0][0] > 0:
            gray = cv2.bitwise_not(img)   
        else:
            gray = np.copy(img)
            
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        vertical   = np.copy(bw)

        verticalsize = int(vertical.shape[0] / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))

        vertical = cv2.erode( vertical, verticalStructure)
        thresh = cv2.dilate( vertical, verticalStructure)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

        cons = []
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            cons.append( (area, x, y, w, h) )

        dino = cons[np.argmax([ a for a, x,y,w,h in cons])]
        cons.remove( dino )
        obs = self.__find_merge_obs(cons)
        
        dino, obs, screen = self.get_distance(dino, obs)

        for x,y,w,h in screen:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # cv2.imshow('image',img)
        # cv2.imshow('image2',thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img, dino, obs


    def __find_board_game(self):
        dino = cv2.imread(os.path.join('Prefabs', 'dino.PNG'), 0)
        h, w = dino.shape

        screen_shot = self.get_screen_shot()
        res = cv2.matchTemplate(screen_shot, dino, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > self.thresold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            self.game_board = (top_left[0], bottom_right[1] - self.board_size[0], bottom_right[1] + self.board_size[1], bottom_right[1])
            self.game_found = True
            self.max_y = top_left[1] - self.game_board[1] + 5


def main():        
    gs = GameScreen()
    if not gs.game_found:
        print('GAME NOT FOUND')
        exit()
        
    screen = gs.get_screen_process()
    lt = time.time()
    pd = 1000
    total = 0
    idx = 1
    while True:
        screen, _, obs = gs.get_screen_process()
        cv2.imshow('image',screen)
        dif_time = time.time() - lt
        if not obs:
            d, a, l = 1000, 0, 0
        else:
            d, l, a = obs[0]
        
        if d > pd:
            total += 1
        else:
            fps = 1/dif_time
            v = (pd - d) * fps

        pd = d
        #print('Score {}, Distancia {}, altura {}, lagura {}, velocidade {}'.format(total,d,a,l,v))
        name = '{} - d {} - a {} - l {}.png'.format( idx, d, a, l)
        print( name )
        cv2.imwrite(name, screen)
        idx += 1
        lt = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()