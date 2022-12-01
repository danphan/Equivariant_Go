from dlgo.gotypes import Point

def is_point_an_eye(board, point, color):
    #point must be empty
    if board.get(point) is not None:
        return False
    #all adjacent points must contain friendly stones (same color)
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False

    #we must control >=3 corners if in the middle of the board (all corners if on edge)
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row-1, point.col -1),
        Point(point.row-1, point.col +1),
        Point(point.row+1, point.col -1),
        Point(point.row+1, point.col +1),
    ]

    for corner in corners:
        #check if corner on board
        if board.is_on_grid(corner):
            #check if corner point has a stone there, and of the correct color
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3
