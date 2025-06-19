import pygame
import numpy as np
import random
import sys
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

# Grid settings
GRID_SIZE = 6
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Q-learning settings
alpha = 0.1
gamma = 0.9
epsilon = 0.3

# Runtime settings
SAVE_INTERVAL = 10
MAX_STEPS = 100
train_mode = True

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
LIGHTBLUE = (173, 216, 230)

# Directions
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# Load or initialize Q-table
if os.path.exists("q_table.npy"):
    Q = np.load("q_table.npy")
else:
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

def is_valid(x, y, maze):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and maze[x][y] != -1

def is_solvable(maze, start, goal):
    visited = [[False]*GRID_SIZE for _ in range(GRID_SIZE)]
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if [x, y] == goal:
            return True
        for d in range(4):
            nx, ny = x + dx[d], y + dy[d]
            if is_valid(nx, ny, maze) and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))
    return False

def create_maze():
    while True:
        maze = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        maze[GRID_SIZE-1][GRID_SIZE-1] = 10
        wall_count = random.randint(5, 10)
        while wall_count > 0:
            x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            if (x, y) not in [(0, 0), (GRID_SIZE-1, GRID_SIZE-1)] and maze[x][y] == 0:
                maze[x][y] = -1
                wall_count -= 1
        if is_solvable(maze, [0, 0], [GRID_SIZE-1, GRID_SIZE-1]):
            return maze

def draw_grid(screen, maze, player_pos, enemy_pos, visited, path=[]):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            if [i, j] == player_pos:
                pygame.draw.rect(screen, BLUE, rect)
            elif [i, j] == enemy_pos:
                pygame.draw.rect(screen, RED, rect)
            elif maze[i][j] == -1:
                pygame.draw.rect(screen, BLACK, rect)
            elif maze[i][j] == 10:
                pygame.draw.rect(screen, GREEN, rect)
            elif [i, j] in path:
                pygame.draw.rect(screen, LIGHTBLUE, rect)
            else:
                pygame.draw.rect(screen, GREY, rect)
            pygame.draw.rect(screen, WHITE, rect, 2)

def enemy_script(player, enemy, maze):
    px, py = player
    ex, ey = enemy
    if abs(px - ex) + abs(py - ey) <= 2:
        if px < ex and is_valid(ex-1, ey, maze): return [ex-1, ey]
        if px > ex and is_valid(ex+1, ey, maze): return [ex+1, ey]
        if py < ey and is_valid(ex, ey-1, maze): return [ex, ey-1]
        if py > ey and is_valid(ex, ey+1, maze): return [ex, ey+1]
    for _ in range(10):
        nx = ex + random.choice([-1, 0, 1])
        ny = ey + random.choice([-1, 0, 1])
        if is_valid(nx, ny, maze):
            return [nx, ny]
    return [ex, ey]

def choose_action(x, y, visited, maze):
    actions = list(range(4))
    random.shuffle(actions)

    valid_actions = []
    for action in actions:
        nx, ny = x + dx[action], y + dy[action]
        if is_valid(nx, ny, maze) and not visited[nx][ny]:
            valid_actions.append(action)

    if random.random() < epsilon and valid_actions:
        return random.choice(valid_actions)
    elif valid_actions:
        return max(valid_actions, key=lambda a: Q[x][y][a])
    else:
        fallback = [a for a in range(4) if is_valid(x + dx[a], y + dy[a], maze)]
        return random.choice(fallback) if fallback else random.randint(0, 3)

def plot_training_graphs(rewards, steps):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, label="Steps", color='orange')
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_graph.png")

def save_q_heatmap(Q):
    heatmap = np.max(Q, axis=2)
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap='YlGnBu', interpolation='nearest')
    plt.colorbar(label='Max Q-Value')
    plt.title("Q-Table Heatmap (Max Action Values)")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            plt.text(j, i, f"{heatmap[i][j]:.2f}", ha='center', va='center', color='black')
    plt.savefig("q_table_heatmap.png")
    plt.close()

def main():
    global Q, train_mode, epsilon
    pygame.init()
    mode = input("Enter 'a' for Auto or 'm' for Manual mode: ").strip().lower()
    if mode == 'm':
        train_mode = False
        epsilon = 0
    else:
        train_mode = True

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Maze Runner AI")
    clock = pygame.time.Clock()
    episode = 0
    reward_history = []
    steps_history = []

    maze = create_maze()

    while True:
        player = [0, 0]
        enemy = [GRID_SIZE-1, 0]
        visited = [[False]*GRID_SIZE for _ in range(GRID_SIZE)]
        done = False
        steps = 0
        total_reward = 0
        path = []
        goal_reached = False

        if not train_mode:
            print("Use arrow keys to move manually. Press ESC to quit.")

        while not done and steps < MAX_STEPS:
            screen.fill(WHITE)
            draw_grid(screen, maze, player, enemy, visited, path)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    np.save("q_table.npy", Q)
                    plot_training_graphs(reward_history, steps_history)
                    save_q_heatmap(Q)
                    pygame.quit()
                    sys.exit()

            x, y = player
            if not train_mode:
                keys = pygame.key.get_pressed()
                action = None
                if keys[pygame.K_UP]: action = 0
                elif keys[pygame.K_DOWN]: action = 1
                elif keys[pygame.K_LEFT]: action = 2
                elif keys[pygame.K_RIGHT]: action = 3
                if action is None:
                    clock.tick(10)
                    continue
            else:
                action = choose_action(x, y, visited, maze)

            nx, ny = x + dx[action], y + dy[action]

            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and is_valid(nx, ny, maze):
                reward = 0
                if maze[nx][ny] == 10:
                    reward = 50
                    done = True
                    goal_reached = True
                    print("🎉 Reached Goal!")
                elif [nx, ny] == enemy:
                    reward = -50
                    done = True
                    print("💀 Caught by enemy!")
                elif visited[nx][ny]:
                    reward = -0.5
                else:
                    reward = -0.1

                reward = float(reward)
                next_action = np.argmax(Q[nx][ny])
                Q[x][y][action] += alpha * (reward + gamma * Q[nx][ny][next_action] - Q[x][y][action])
                player = [nx, ny]
                visited[nx][ny] = True
                path.append([nx, ny])

                if reward == 50:
                    np.save("q_table.npy", Q)
                    save_q_heatmap(Q)
                    print("Q-table and heatmap saved after goal reached.")
                    maze = create_maze()
                    break
            else:
                reward = -1
                reward = float(reward)
                Q[x][y][action] += alpha * (reward + gamma * np.max(Q[x][y]) - Q[x][y][action])

            enemy = enemy_script(player, enemy, maze)
            if player == enemy:
                Q[x][y][action] += alpha * (-50 - Q[x][y][action])
                done = True
                print("💀 Caught by enemy!")

            steps += 1
            total_reward += reward
            clock.tick(5 if train_mode else 10)

        if goal_reached:
            episode += 1

        reward_history.append(total_reward)
        steps_history.append(steps)

        if total_reward < 50:
            if episode % 5 == 0 and episode > 0:
                np.save("q_table.npy", Q)
                save_q_heatmap(Q)
                print(f"Episode {episode} Q-table and heatmap saved (interval).")

        if train_mode:
            epsilon = max(0.05, epsilon * 0.97)

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            plot_training_graphs(reward_history, steps_history)

        print(f"Episode {episode} | Steps: {steps} | Total Reward: {round(total_reward, 2)}\n")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
