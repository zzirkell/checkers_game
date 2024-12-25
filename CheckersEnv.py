import numpy as np


def initialize_board():
    """Initializes the 6x6 board"""
    board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [-1, 0, -1, 0, -1, 0],
                      [0, -1, 0, -1, 0, -1]])
    return board


class CheckersEnv:

    def __init__(self):
        """Initialization"""
        self.board = initialize_board()
        self.player = 1

    def reset(self):
        """Resets the board"""
        self.board = initialize_board()
        self.player = 1

    def step(self, action, player):
        """Updates the board according to the action and calculates the reward for the agent"""
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

        if end_row == 5:
            reward += 10

        if player == 1:
            if end_row + 1 < 6:
                next_row = self.board[end_row + 1]
                if end_col + 1 < 6:
                    if next_row[end_col + 1] == -player or next_row[end_col + 1] == -2 * player:
                        reward -= 50
                if end_col - 1 > 0:
                    if next_row[end_col - 1] == -player or next_row[end_col - 1] == -2 * player:
                        reward -= 50
        if player == -1:
            if end_row - 1 > 0:
                next_row = self.board[end_row - 1]
                if end_col + 1 < 6:
                    if next_row[end_col + 1] == -player or next_row[end_col + 1] == -2 * player:
                        reward -= 50
                if end_col - 1 > 0:
                    if next_row[end_col - 1] == -player or next_row[end_col - 1] == -2 * player:
                        reward -= 50

        winner = self.check_game_winner()

        if winner == player:
            reward += 100
        elif winner == -player:
            reward -= 100

        if not self.get_valid_moves(player):
            reward = -100
        if not self.get_valid_moves(-player):
            reward = 100

        done = winner != 0

        return self.board, reward, done

    def get_valid_moves(self, player):
        """Returns all valid moves for each piece"""
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
                            if (self.board[r + dr][c + dc] == -player or self.board[r + dr][c + dc] == -2 * player) and \
                                    self.board[nr][nc] == 0:
                                return [[r, c, nr, nc]]
        return moves

    def get_valid_moves_for_piece(self, piece, player):
        """Returns all valid moves for one piece"""
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

    def check_captured(self, action):
        """Checks if there was a piece captured by the action"""
        start_row, start_col, end_row, end_col = action
        dr = (end_row - start_row) // abs(end_row - start_row)
        dc = (end_col - start_col) // abs(end_col - start_col)

        if abs(end_row - start_row) == 2:
            self.board[start_row + dr][start_col + dc] = 0

    def check_game_winner(self):
        """Checks if the game is over and determines the winner"""
        if np.sum(self.board < 0) == 0:
            return 1
        elif np.sum(self.board > 0) == 0:
            return -1
        elif len(self.get_valid_moves(-1)) == 0:
            return 1
        elif len(self.get_valid_moves(1)) == 0:
            return -1
        return 0
