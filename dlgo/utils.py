from dlgo import gotypes
import numpy as np

"""
Definitions to help printing out current board position
"""
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ', 
    gotypes.Player.white: ' o ',
}


"""
Prints out move to screen
"""
def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '{}{}'.format(COLS[move.point.col-1], move.point.row)
    print('{} {}'.format(player,move_str))

def print_board(board):
    #print out the rows from top to bottom
    for row in range(board.num_rows, 0, -1): #looping backwards since we want top to bottom
        bump = ' ' if row <= 9 else ''
        #loop through the columns in a given row
        line = []
        for col in range(1,board.num_cols+1):
            #get point at row, col
            stone = board.get(gotypes.Point(row,col))
            line.append(STONE_TO_CHAR[stone])
        print('{}{} {}'.format(bump,row,''.join(line))) #note that row labels have been added
        
    #after all rows have been printed out, add column labels
    print('    ' + '  '.join(COLS[:board.num_cols]))

"""
Turns human-readable coordinate into actual Point
"""
def point_from_coords(coords):
    col = COLS.index(coords[0])+1
    row = int(coords[1:]) #colon needed since row can have 2 digits
    return gotypes.Point(row=row,col=col)

# NOTE: MoveAge is only used in chapter 13, and doesn't make it to the main text.
# This feature will only be implemented in goboard_fast.py so as not to confuse
# readers in early chapters.
class MoveAge():
    def __init__(self, board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1
