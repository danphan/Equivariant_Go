import importlib

class Encoder:
    """
    Lets us support logging/save the name of the encoder our model is using
    """
    def name(self):
        raise NotImplementedError()

    """
    Turns Go Board into numerical data
    """
    def encode(self, game_state):
        raise NotImplementedError()

    """
    Turns Go board point into integer index
    """
    def encode_point(self, point):
        raise NotImplementedError()

    """
    Turns integer index back into Go board point
    """
    def decode_point_index(self, index):
        raise NotImplementedError()

    """
    Number of points on the board: num_rows x num_cols
    """
    def num_points(self):
        raise NotImplementedError()

    """
    Shape of the encoded board structure
    """
    def shape(self):
        raise NotImplementedError()

"""
Convenience function which allows us to create encoders by name
"""
def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module('dlgo.encoders.'+name)
    #each encoder implementation must have a 'create' function associated
    #with the encoder class definition which provides an instance
    constructor = getattr(module, 'create')
    return constructor(board_size)
