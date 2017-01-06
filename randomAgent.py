from  ple.games.monsterkong import MonsterKong
from ple import PLE
import  random


moves = [ord('w'), ord('a'), ord ('s'), ord('d'),
         ord(' ')]
game = MonsterKong()
p = PLE(game, fps=30, display_screen=True)

p.init()
reward = 0.0

for i in range(100000):
   if p.game_over():
           p.reset_game()
           break

   observation = p.getScreenRGB()
   reward = p.act(random.choice(moves))