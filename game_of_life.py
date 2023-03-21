import pygame
import numpy as np

class GameOfLife:
    def __init__(self, width, height, cell_size, predictive_function):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = np.random.choice([0, 1], (height, width))
        self.predictive_function = predictive_function
        self.score = 0

    # Game of Life logic
    def step_logic(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                total = np.sum(self.grid[i - 1:i + 2, j - 1:j + 2]) - self.grid[i, j]
                if self.grid[i, j] and (total == 2 or total == 3):
                    new_grid[i, j] = 1
                elif not self.grid[i, j] and total == 3:
                    new_grid[i, j] = 1
        self.grid = new_grid

    def step(self):
        predicted_grid = self.predictive_function(self.grid)
        predicted_grid = np.array(predicted_grid)

        if predicted_grid.shape != self.grid.shape:
            raise ValueError("Predictive function returned grid of incorrect shape")

        self.step_logic()

        self.score += np.sum(self.grid == predicted_grid)

    def draw(self, surface):
        for y in range(self.height):
            for x in range(self.width):
                color = (255, 255, 255) if self.grid[y, x] else (0, 0, 0)
                pygame.draw.rect(surface, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))


def predictive_function_example(previous_grid):
    # Replace this with your predictive function.
    # For now, it just returns the input unchanged.
    return previous_grid

if __name__ == "__main__":
    pygame.init()
    WIDTH, HEIGHT = 500, 500
    CELL_SIZE = 50

    game_of_life = GameOfLife(WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE, CELL_SIZE, predictive_function_example)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Conway's Game of Life")

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        game_of_life.step()
        screen.fill((0, 0, 0))
        game_of_life.draw(screen)
        pygame.display.flip()

        print(f"Current accuracy score: {game_of_life.score}")

        clock.tick(1)  # Update once per second

    pygame.quit()
