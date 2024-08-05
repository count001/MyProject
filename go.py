import numpy as np
import torch
import time
from sgfmill import sgf


class Go:
    # Set the default size of the chessboard to 19 * 19
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        # Liberty means that the initial degree of freedom for each position is 0
        self.liberty = np.zeros((size, size), dtype=np.int8)
        # Save the previous state of the chessboard
        self.previousBoard = np.zeros((size, size), dtype=np.int8)
        # Create a list of length 8, with each element being (None, None), representing the history of the last 8 moves.
        self.history = [(None, None)] * 8

    # clone
    def clone(self):
        go = Go(self.size)
        go.board = np.array(self.board)
        go.liberty = np.array(self.liberty)
        go.previousBoard = np.array(self.previousBoard)
        go.history = list(self.history)
        return go

    def move(self, color, x, y):
        # Check if the position of the drop (x, y) is within the range of the chessboard.
        # If not present, returning False indicates that the position is invalid.
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False

        # 1. Check if there are already chess pieces
        if self.board[x, y] != 0:
            return False

        # 2. Check for robber
        # Robbery refers to the situation where the previous move results in the current (x, y) position being empty
        # and the position about to be replaced is surrounded by the opponent's pieces.
        # This position is considered a robbery and no placement is allowed
        anotherColor = -color
        if self.history[-2] == (x, y):
            if (x == 0 or self.board[x - 1, y] == anotherColor) and \
               (y == 0 or self.board[x, y - 1] == anotherColor) and \
               (x == self.size - 1 or self.board[x + 1, y] == anotherColor) and \
               (y == self.size - 1 or self.board[x, y + 1] == anotherColor):
                return False

        # 3. Remove chess pieces without freedom
        self.previousBoard = np.array(self.board)
        self.board[x, y] = color
        if x > 0:
            self.clearColorNear(anotherColor, x - 1, y)
        if x < self.size - 1:
            self.clearColorNear(anotherColor, x + 1, y)
        if y > 0:
            self.clearColorNear(anotherColor, x, y - 1)
        if y < self.size - 1:
            self.clearColorNear(anotherColor, x, y + 1)

        self.clearColorNear(color, x, y)

        # If after placing a chess piece, it is found that the piece you placed does not have freedom,
        # then the piece is removed, indicating that this move is suicide (illegal)
        # Need to restore the previous chessboard state, returning False indicates that the drop is invalid.
        if self.board[x, y] == 0:
            self.board = self.previousBoard
            return False

        # Add the position of the drop (x, y) to the history record
        self.history.append((x, y))

        return True

    def clearColorNear(self, color, x, y):
        if self.board[x, y] != color:
            return

        # The auxiliary arrays visited and boardGroup are used to mark the visited positions and the current position of the chess piece group,
        # both of which are the size of the chessboard and have an initial value of 0.
        visited = np.zeros((self.size, self.size), dtype=np.int32)
        boardGroup = np.zeros((self.size, self.size), dtype=np.int32)

        # Define a depth first search (DFS) function to traverse the current chess piece group and its surrounding positions.
        def dfs(colorBoard, x, y):
            if visited[x, y] == 1:
                return
            # If the current position (x, y) is not a chess piece of the specified color, check if it is an empty position.
            # If it is an empty position, add it to the allLibertyPosition set, indicating that the position is a liberty, and then return it.
            if colorBoard[x, y] == 0:
                if self.board[x, y] == 0:
                    allLibertyPosition.add((x, y))
                return
            visited[x, y] = 1
            boardGroup[x, y] = 1

            # Recursively call DFS functions to traverse adjacent positions up, down, left, and right.
            if x > 0:
                dfs(colorBoard, x - 1, y)
            if x < self.size - 1:
                dfs(colorBoard, x + 1, y)
            if y > 0:
                dfs(colorBoard, x, y - 1)
            if y < self.size - 1:
                dfs(colorBoard, x, y + 1)

        colorBoard = self.board == color

        allLibertyPosition = set()
        dfs(colorBoard, x, y)

        # Calculate the amount of freedom in the current chess piece group.
        # If the number of liberties is 0, it means that the chess piece group is empty, marked as dead and removed from the board.
        # Otherwise, update the number of liberties for the chess piece group.
        liberties = len(allLibertyPosition)
        # dead group
        if liberties == 0:
            self.board[boardGroup == 1] = 0
        else:
            self.liberty[boardGroup == 1] = liberties


def toDigit(x, y):
    return x * 19 + y


def toPosition(digit):
    if isinstance(digit, torch.Tensor):
        digit = digit.item()

    if digit == 361:
        return None, None
    x = digit // 19
    y = digit % 19
    return x, y


def testKill():
    go = Go()
    #     3 4 5 6
    # 15    B W
    # 16  B W B W
    # 17    B W
    go.move(1, 15, 4)
    assert go.move(2, 15, 4) == False

    go.move(-1, 15, 5)
    go.move(1, 16, 3)
    go.move(-1, 16, 4)
    go.move(1, 17, 4)
    go.move(-1, 17, 5)
    go.move(1, 16, 5)
    go.move(-1, 16, 6)

    print(go.board)

    assert go.board[16, 4] == 0

    go.move(1, 4, 4)
    go.move(-1, 16, 4)

    assert go.move(1, 16, 4) == False

    print(go.board)


def testLiberty():
    go = Go()
    go.move(1, 4, 4)
    go.move(1, 4, 5)
    go.move(1, 4, 6)
    go.move(1, 5, 4)
    go.move(-1, 5, 5)

    print(go.board)
    print(go.liberty)

    assert go.liberty[4, 4] == go.liberty[4, 5] == go.liberty[4, 6] == go.liberty[5, 4] == 8
    assert go.liberty[5, 5] == 2


def testTime():
    with open('test.sgf', 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)
    # for move in validSequence:
    #     print(move)
    go = Go()

    start = time.time()
    for move in validSequence:
        if move[0] == 'w':
            color = 1
        else:
            color = 2
        x = move[1][0]
        y = move[1][1]
        go.move(color, x, y)
        # print(go.board)
    end = time.time()
    print('time:', end - start)


if __name__ == '__main__':
    testKill()
    testLiberty()
    testTime()
