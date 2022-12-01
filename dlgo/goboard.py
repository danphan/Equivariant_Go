import copy
from dlgo.gotypes import Player
from dlgo import zobrist

#a class for implementing a move (play on a point, pass, or resign)
class Move:
    def __init__(self, point = None, is_pass = False, is_resign = False):
        #a move can only be playing on a point XOR passing XOR resigning
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass = True)

    @classmethod
    def is_resign(cls):
        return Move(is_resign = True)

class GoString:
    """
    Class to represent a string of connected stones.

    Attributes:
    -----------
    color: Player enum
    stones: set of Points
    liberties: set of Points

    Methods:
    --------
    
    """
    def __init__(self, color, stones, liberties):
        self.color = color

        #make stones and liberties immutable
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    #replace remove_liberty with without_liberty to accomodate the immutability of stones and liberties
    #now instead of simply updating GoString.liberties, we return a new GoString
    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)
    
    #replace add_liberty with with_liberty to accomodate the immutability of stones and liberties
    #now instead of simply updating GoString.liberties, we return a new GoString
    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones
        )

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties

class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        #private dictionary used to keep track of GoStrings (keys = points, values = GoStrings)
        self._grid = {} 

        #hash to store board state
        self._hash = zobrist.EMPTY_BOARD

    """
    After placing stone, we
    1. want to merge any adjacent strings of the same color
    2. reduce liberties of any adjacent strings of opposite color
    3. remove any opposite-color strings which now have zero liberties
    """
    def place_stone(self, player, point):
        assert self.is_on_grid(point) #make sure desired point is on the board
        assert self._grid.get(point) is None #make sure point is empty before placing stone
        adjacent_same_color = [] #list of GoStrings adjacent to point of same color
        adjacent_opposite_color = [] #list of GoStrings adjacent to point of opposite color
        liberties = [] #liberties of stone placed at point

        for neighbor in point.neighbors():
            #make sure point is actually on board
            if not self.is_on_grid(neighbor):
                continue
            #see if there is a GoString attached to this point
            neighbor_string = self._grid.get(neighbor) 

            #if point is empty, mark it as a liberty of the just-placed stone
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
            
        #create new GoString for stone at point (to be merged with adjacent GoStrings of same color)
        new_string = GoString(player, [point], liberties) 

        #merge adjacent strings of the same color
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)

        #add new string to _grid
        for stone in new_string.stones:
            self._grid[stone] = new_string

        #modify hash of board
        self._hash ^= zobrist.HASH_CODE[point, player]

        #reduce liberties of opposite-color strings adjacent to point
        for opposite_color_string in adjacent_opposite_color:
            reduced_string = opposite_color_string.without_liberty(point)

            #remove any dead opposite-color strings from grid
            if reduced_string.num_liberties == 0:
                self._remove_string(opposite_color_string)

    """
    Getter for zobrist hash
    """
    def zobrist_hash(self):
        return self._hash        
            
    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    """
    Returns the content of the Board at a point. 
    a Player(black or white) if a stone is there, None otherwise
    """
    def get(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    """
    Returns the GoString associated with a point. 
    """
    def get_go_string(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    """
    Updates string from self._grid dictionary (does not worry about neighboring strings)
    Used as a helper function for _remove_string below
    """
    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    """
    Remove string (making sure to add appropriate liberties to neighbors
    """
    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():
                #check if neighbor has associated string
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                #update _grid to have the updated neighbor string with the extra liberty
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            self._grid[point] = None

            #update board hash after deleting stone
            self._hash ^= zobrist.HASH_CODE[point, string.color]

"""
GameState contains info regarding
1. Current Board position
2. Next Player (i.e. which player's turn)
3. previous GameState
4. move that was just played
"""
class GameState:

    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous #store previous board state here

        """
        self.previous_states is a frozenset of all previous states (using zobrist hashing)
        note that we include the player info
        otherwise, an empty board with white going first would be the same as black going first
        i.e. hash collision
        """
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})
        self.last_move = move

    """
    Applies move and returns new GameState
    """
    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)

        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    """
    Create new game from input board size
    """
    @classmethod
    def new_game(cls, board_size):
        assert isinstance(board_size, int)
        board_size = (board_size, board_size)
        board = Board(*board_size) 
        return cls(board, Player.black, None, None) #cls might as well be GameState

    """
    Check if a game is over.
    (This happens when both players pass.)
    """
    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass 

    """
    Check if a move leads to self-capture
    """
    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point) 
        #recall that place_stone plays the move and returns the updated board
        #if after this move is played, the string associated with point has zero liberties, then this move was self-capture
        new_string = next_board.get_go_string(move.point)
        if new_string is None:
            return False
        return (new_string.num_liberties == 0)
        

    """
    Check if current GameState violates situational superko rule (if current game state recreates one of the previous game states, including stones on the board and whose turn it is)
    """
    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False    
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return (next_situation in self.previous_states) #loops through the frozenset to check superko

    """
    Check if a move is valid
    """
    def is_valid_move(self, move):
        #if game is over, no more moves can be played
        if self.is_over():
            return False
        #if game is not over, one is always allowed to pass or resign
        if move.is_pass or move.is_resign:
            return True
        #otherwise, move (i) must be played on empty point, (ii) cannot be self-capture, (iii) cannot violate ko
        return (self.board.get(move.point) is None) and \
                (not self.is_move_self_capture(self.next_player, move)) and \
                (not self.does_move_violate_ko(self.next_player, move))
