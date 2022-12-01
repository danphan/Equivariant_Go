from dlgo.agent.naive import RandomBot
from dlgo import goboard_slow #cannot use goboard_fast unless we use 19x19
from dlgo.gotypes import Player
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input

def main():
    board_size = 9
    game = goboard_slow.GameState.new_game(board_size)
    bot = RandomBot()

    while not game.is_over():

    
        #clear the screen before each new move
        print(chr(27) + "[2J") 

        print_board(game.board)
        
        if game.next_player == Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard_slow.Move.play(point)
        else:
            
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)        


if __name__ == '__main__':
    main()
