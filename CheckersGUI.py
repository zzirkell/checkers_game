import pygame
import sys

class CheckersGUI:
    def __init__(self, env, agent):
        pygame.init()
        self.env = env
        self.agent = agent
        self.window_size = 600
        self.square_size = self.window_size // 6
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("6x6 Checkers Game")
        self.running = True
        self.selected_piece = None
        self.valid_moves = []
        self.clock = pygame.time.Clock()
        self.agent_turn()
        self.run_game()

    def draw_board(self):
        colors = [(209, 139, 71), (255, 206, 158)]
        for row in range(6):
            for col in range(6):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.window, color, (col * self.square_size, row * self.square_size, self.square_size, self.square_size))
                piece = self.env.board[row][col]
                if piece != 0:
                    self.draw_piece(row, col, piece)
                if self.selected_piece:
                    self.draw_valid_moves()

    def draw_piece(self, row, col, piece):
        center = (col * self.square_size + self.square_size // 2, row * self.square_size + self.square_size // 2)
        radius = self.square_size // 3
        color = (255, 255, 255) if piece > 0 else (0, 0, 0)
        pygame.draw.circle(self.window, color, center, radius)
        if abs(piece) > 1:
            pygame.draw.circle(self.window, (255, 0, 0), center, radius // 2)

    def run_game(self):
        while self.running:
            self.draw_board()
            self.handle_events()
            pygame.display.flip()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.on_click(event.pos)

    def on_click(self, pos):
        col, row = pos[0] // self.square_size, pos[1] // self.square_size
        if self.selected_piece:
            self.make_move(row, col)
        else:
            self.select_piece(row, col)

    def select_piece(self, row, col):
        if self.env.board[row][col] == -1 or self.env.board[row][col] == -2:
            self.selected_piece = (row, col)
            self.valid_moves = [move for move in self.env.get_valid_moves_for_piece(self.selected_piece, -1) if move[:2] == [row, col]]

    def draw_valid_moves(self):
        for move in self.valid_moves:
            row, col = move[2:]
            center = (col * self.square_size + self.square_size // 2, row * self.square_size + self.square_size // 2)
            radius = self.square_size // 3
            color = (0, 255, 0)
            pygame.draw.circle(self.window, color, center, radius // 4)

    def make_move(self, row, col):
        move = [self.selected_piece[0], self.selected_piece[1], row, col]
        if move in self.valid_moves:
            self.env.step(move, -1)
            self.selected_piece = None
            self.valid_moves = []
            self.check_game_state()
            self.agent_turn()
        else:
            self.selected_piece = None
            self.valid_moves = []

    def agent_turn(self):
        state = self.env.board.flatten()
        valid_moves = self.env.get_valid_moves(1)
        action = self.agent.select_action(state, valid_moves)
        self.env.step(action, 1)
        self.check_game_state()

    def check_game_state(self):
        winner = self.env.check_game_winner()
        if winner != 0:
            print("GAME OVER")
            if winner == 1:
                print("YOU LOST!")
            else:
                print("YOU WON!")
            self.env.reset()