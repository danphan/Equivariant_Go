import enum
from collections import namedtuple

"""
Define a set of immutable constants using enum.
Can access constants using Player.black, Player['black'], or Player('black')

The existence of Player('black') means that if you write:
a = Player()
...you'll get an error

Think about black and white as class attributes, which can be accessed without instantiation.
"""
class Player(enum.Enum):

    #set possible values for Player class
    black = 1
    white = 2

    #switch between black and white by calling Player.other
    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

"""
define a Point (sub)class which has a row and column (type of named tuple)
but also has neighbors

Can obtain a point's row and col using Point.row and Point.col
"""
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col-1),
            Point(self.row, self.col+1),
        ]

