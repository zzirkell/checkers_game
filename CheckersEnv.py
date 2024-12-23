import numpy as np


def initialize_board():
    board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [-1, 0, -1, 0, -1, 0],
                      [0, -1, 0, -1, 0, -1]])
    return board


class CheckersEnv:

    def __init__(self):
        self.board = initialize_board()
        self.player = 1

    def reset(self):
        self.board = initialize_board()
        self.player = 1

    def get_valid_moves_for_piece(self, piece, player):
        moves = []
        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]

        row, col = piece

        if self.board[row][col] == 2 * player:
            directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 6 and 0 <= nc < 6 and self.board[nr][nc] == 0:
                moves.append([row, col, nr, nc])

            nr, nc = row + 2 * dr, col + 2 * dc
            if 0 <= nr < 6 and 0 <= nc < 6:
                blocking_piece = self.board[row + dr][col + dc]
                if (blocking_piece == -player or blocking_piece == -player * 2) and self.board[nr][nc] == 0:
                    moves.append([row, col, nr, nc])

        return moves

    def get_valid_moves(self, player):
        moves = []

        for r in range(6):
            for c in range(6):
                if self.board[r][c] == player or self.board[r][c] == 2 * player:
                    if self.board[r][c] == 2 * player:
                        directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
                    else:
                        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 6 and 0 <= nc < 6 and self.board[nr][nc] == 0:
                            moves.append([r, c, nr, nc])
                        nr, nc = r + 2 * dr, c + 2 * dc
                        if 0 <= nr < 6 and 0 <= nc < 6:
                            if self.board[r + dr][c + dc] == -player and self.board[nr][nc] == 0:
                                moves.append([r, c, nr, nc])
        return moves

    def check_captured(self, action):
        start_row, start_col, end_row, end_col = action
        dr = (end_row - start_row) // abs(end_row - start_row)
        dc = (end_col - start_col) // abs(end_col - start_col)

        if abs(end_row - start_row) == 2:
            self.board[start_row + dr][start_col + dc] = 0

    def check_game_winner(self):
        if np.sum(self.board < 0) == 0:
            return 1
        elif np.sum(self.board > 0) == 0:
            return -1
        elif len(self.get_valid_moves(-1)) == 0:
            return 1
        elif len(self.get_valid_moves(1)) == 0:
            return -1
        return 0

    def step(self, action, player):
        start_row, start_col, end_row, end_col = action

        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0

        self.check_captured(action)

        if player == 1 and end_row == 5:
            self.board[end_row][end_col] = 2
        elif player == -1 and end_row == 0:
            self.board[end_row][end_col] = -2

        reward = 0
        if abs(end_row - start_row) == 2:
            reward = 10

        winner = self.check_game_winner()

        if winner == player:
            reward += 100
        elif winner == -player:
            reward -= 100

        return self.board, reward

    def render(self):
        for row in self.board:
            for square in row:
                if square == 1:
                    piece = "|0"
                elif square == -1:
                    piece = "|X"
                elif square == 2:
                    piece = "|K"
                elif square == -2:
                    piece = "|Q"
                else:
                    piece = "| "
                print(piece, end='')
            print("|")
