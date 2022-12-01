import random
from dlgo.agent.base import Agent
from dlgo.gotypes import Point
from dlgo.goboard_slow import Move
from dlgo.agent.helpers import is_point_an_eye

class RandomBot(Agent):
    def select_move(self, game_state):
        """
        Choose a random move that preserves our own eyes.
        """
        candidates = []
        #loop through all points of board
        for r in range(1,game_state.board.num_rows+1):
            for c in range(1,game_state.board.num_cols+1):
                #check if point deletes one of our eyes
                candidate = Point(row=r,col=c)
                if (not is_point_an_eye(game_state.board, 
                                        candidate, 
                                        game_state.next_player)) and \
                    game_state.is_valid_move(Move.play(candidate)):
                    candidates.append(candidate)
        
        if candidates:
            return Move.play(random.choice(candidates))
        return Move.pass_turn()
