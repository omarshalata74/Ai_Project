import pygame
import sys
import random
import copy
import heapq
from collections import deque

pygame.init()

WIDTH, HEIGHT = 600, 460
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man AI v3 - Smart & Safe")

BLACK  = (0,   0,   0  )
BLUE   = (0,   0,   200)
YELLOW = (255, 255, 0  )
WHITE  = (255, 255, 255)
RED    = (220, 0,   0  )
GREEN  = (0,   220, 80 )
ORANGE = (255, 165, 0  )

# ═══════════════════════════════════════════════════════════════════
#  MAZE
# ═══════════════════════════════════════════════════════════════════
ORIGINAL_MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,0,1,1,0,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

ROWS      = len(ORIGINAL_MAZE)
COLS      = len(ORIGINAL_MAZE[0])
TILE_SIZE = WIDTH // COLS

font_big = pygame.font.SysFont(None, 48)
font_med = pygame.font.SysFont(None, 32)
clock    = pygame.time.Clock()

STUCK_THRESHOLD = 8   # ticks in a small area before ghost is considered stuck

# ═══════════════════════════════════════════════════════════════════
#  GAME STATE
# ═══════════════════════════════════════════════════════════════════
maze        = []
player_pos  = [1, 1]
ghosts      = []
score       = 0
visited     = {}   # (r,c) -> int  visit count

def reset_game():
    global maze, player_pos, ghosts, score, visited
    maze       = copy.deepcopy(ORIGINAL_MAZE)
    player_pos = [1, 1]
    visited    = {}
    score      = 0
    ghosts = [
        {"pos": [13, 10], "dir": (0, 1),  "mode": "random", "timer": 0,
         "pos_history": deque(maxlen=STUCK_THRESHOLD), "unstuck_path": []},
        {"pos": [7,  18], "dir": (1, 0),  "mode": "random", "timer": 0,
         "pos_history": deque(maxlen=STUCK_THRESHOLD), "unstuck_path": []},
    ]

reset_game()

DIRS = [(0,1),(1,0),(0,-1),(-1,0)]

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def walkable(r, c):
    return in_bounds(r, c) and maze[r][c] != 1

def neighbors(pos):
    r, c = pos
    return [(r+dr, c+dc) for dr, dc in DIRS if walkable(r+dr, c+dc)]

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ═══════════════════════════════════════════════════════════════════
#  COLLISION  — the RIGHT way
#  Check if Pacman and any ghost occupy the same cell OR
#  if they SWAPPED cells this tick (pass-through bug fix)
# ═══════════════════════════════════════════════════════════════════
def collision_check(old_player, new_player, old_ghosts_pos, new_ghosts_pos):
    np = tuple(new_player)
    op = tuple(old_player)
    for i, g_new in enumerate(new_ghosts_pos):
        g_old = old_ghosts_pos[i]
        # Same cell
        if np == tuple(g_new):
            return True
        # Swapped cells (passed through each other)
        if np == tuple(g_old) and op == tuple(g_new):
            return True
    return False

# ═══════════════════════════════════════════════════════════════════
#  BFS — plain (used by ghosts to chase)
# ═══════════════════════════════════════════════════════════════════
def bfs(start, goal):
    if start == goal:
        return [start]
    q    = deque([(start, [start])])
    seen = {start}
    while q:
        cur, path = q.popleft()
        for n in neighbors(cur):
            if n not in seen:
                if n == goal:
                    return path + [n]
                seen.add(n)
                q.append((n, path + [n]))
    return []

# ═══════════════════════════════════════════════════════════════════
#  GHOST FUTURE POSITIONS  — predict where ghosts will be in N steps
# ═══════════════════════════════════════════════════════════════════
PREDICT_STEPS = 4   # how many ticks ahead we predict ghosts

def predict_ghost_cells(steps=PREDICT_STEPS):
    """Return a set of cells ghosts might occupy in the next `steps` ticks."""
    danger = set()
    for g in ghosts:
        cur = tuple(g["pos"])
        frontier = {cur}
        for _ in range(steps):
            next_frontier = set()
            for pos in frontier:
                for n in neighbors(pos):
                    next_frontier.add(n)
            frontier |= next_frontier
            danger   |= frontier
    return danger

# ═══════════════════════════════════════════════════════════════════
#  A*  — ghost-aware, with predicted danger zones
# ═══════════════════════════════════════════════════════════════════
def ghost_cost(pos, danger_cells):
    """Penalty for a cell based on ghost proximity AND predicted path."""
    t = tuple(pos)
    penalty = 0
    for g in ghosts:
        d = manhattan(t, tuple(g["pos"]))
        if d == 0:   return 50000   # on top of ghost → never
        if d == 1:   penalty += 5000
        elif d == 2: penalty += 500
        elif d == 3: penalty += 50
    if t in danger_cells:
        penalty += 200
    return penalty

def astar(start, goal, danger_cells, use_penalty=True):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came   = {}
    g_cost = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for n in neighbors(cur):
            step = g_cost[cur] + 1
            if n not in g_cost or step < g_cost[n]:
                g_cost[n] = step
                gp = ghost_cost(n, danger_cells) if use_penalty else 0
                rv = min(visited.get(n, 0) * 5, 40)   # mild revisit penalty
                f  = step + manhattan(n, goal) + gp + rv
                heapq.heappush(open_set, (f, n))
                came[n] = cur
    return []

# ═══════════════════════════════════════════════════════════════════
#  FOOD TARGETING
#  Score each dot: penalise dots that are close to ghosts
# ═══════════════════════════════════════════════════════════════════
def all_dots():
    return [(r, c) for r in range(ROWS) for c in range(COLS) if maze[r][c] == 0]

def best_targets(danger_cells, n=10):
    dots = all_dots()
    pp   = tuple(player_pos)
    def key(d):
        dist  = manhattan(pp, d)
        # bonus: dots far from ghosts are more attractive
        g_far = min(manhattan(d, tuple(g["pos"])) for g in ghosts)
        danger_bonus = 30 if d in danger_cells else 0
        return dist - g_far * 0.8 + danger_bonus
    dots.sort(key=key)
    return dots[:n]

# ═══════════════════════════════════════════════════════════════════
#  ESCAPE  — maximise distance from all ghosts
# ═══════════════════════════════════════════════════════════════════
def escape_move():
    pp   = tuple(player_pos)
    best = None
    best_score = -1
    for n in neighbors(pp):
        # Sum of distances to all ghosts — higher is safer
        total_dist = sum(manhattan(n, tuple(g["pos"])) for g in ghosts)
        # Prefer unvisited cells when escaping
        vis_pen = visited.get(n, 0) * 2
        s = total_dist - vis_pen
        if s > best_score:
            best_score = s
            best = n
    return best

def min_ghost_dist(pos=None):
    if pos is None:
        pos = tuple(player_pos)
    return min(manhattan(pos, tuple(g["pos"])) for g in ghosts)

# ═══════════════════════════════════════════════════════════════════
#  GHOST MOVEMENT — smarter chase with cooldown + anti-stuck logic
# ═══════════════════════════════════════════════════════════════════
CHASE_INTERVAL_MIN = 6
CHASE_INTERVAL_MAX = 15

def _bfs_farthest(start):
    """BFS flood-fill from start; return the farthest reachable cell."""
    visited_bfs = {start}
    q = deque([start])
    last = start
    while q:
        cur = q.popleft()
        last = cur
        for n in neighbors(cur):
            if n not in visited_bfs:
                visited_bfs.add(n)
                q.append(n)
    return last

def move_ghosts():
    for g in ghosts:
        g["timer"] += 1
        r, c = g["pos"]

        # Track position history for stuck detection
        g["pos_history"].append((r, c))

        # ── ANTI-STUCK: force ghost away if it's looping in a small area ──
        is_stuck = (
            len(g["pos_history"]) == STUCK_THRESHOLD and
            len(set(g["pos_history"])) <= 3
        )

        if is_stuck or g["unstuck_path"]:
            if not g["unstuck_path"]:
                # Build a path to the farthest reachable cell to break the loop
                far  = _bfs_farthest((r, c))
                path = bfs((r, c), far)
                g["unstuck_path"] = path[1:min(len(path), STUCK_THRESHOLD + 2)]
                g["pos_history"].clear()

            if g["unstuck_path"]:
                nxt = g["unstuck_path"].pop(0)
                g["pos"] = list(nxt)
                g["dir"] = (nxt[0] - r, nxt[1] - c)
                continue

        # ── Normal: decide chase or random walk ────────────────────────────
        chase_now = g["timer"] >= random.randint(CHASE_INTERVAL_MIN, CHASE_INTERVAL_MAX)
        if chase_now:
            g["timer"] = 0
            path = bfs((r, c), tuple(player_pos))
            if len(path) > 1:
                g["pos"] = list(path[1])
                g["dir"] = (path[1][0] - r, path[1][1] - c)
                continue

        # Random walk — prefer not to reverse direction
        dy, dx  = g["dir"]
        options = [d for d in DIRS if not (d[0] == -dy and d[1] == -dx)]
        random.shuffle(options)
        moved   = False
        for d in options:
            nr, nc = r + d[0], c + d[1]
            if walkable(nr, nc):
                g["pos"] = [nr, nc]
                g["dir"] = d
                moved = True
                break

        if not moved:  # dead end — try all directions including reverse
            all_dirs = DIRS[:]
            random.shuffle(all_dirs)
            for d in all_dirs:
                nr, nc = r + d[0], c + d[1]
                if walkable(nr, nc):
                    g["pos"] = [nr, nc]
                    g["dir"] = d
                    break

# ═══════════════════════════════════════════════════════════════════
#  PLAYER MOVEMENT
# ═══════════════════════════════════════════════════════════════════
def move_player(step):
    global score
    player_pos[0], player_pos[1] = step
    pos = tuple(step)
    visited[pos] = visited.get(pos, 0) + 1
    if maze[step[0]][step[1]] == 0:
        maze[step[0]][step[1]] = 2
        score += 10

# ═══════════════════════════════════════════════════════════════════
#  WIN / DRAW
# ═══════════════════════════════════════════════════════════════════
def check_win():
    return not any(maze[r][c] == 0 for r in range(ROWS) for c in range(COLS))

def draw():
    screen.fill(BLACK)
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if maze[r][c] == 1:
                pygame.draw.rect(screen, BLUE, rect)
                # subtle wall border
                pygame.draw.rect(screen, (0, 0, 120), rect, 1)
            elif maze[r][c] == 0:
                pygame.draw.circle(screen, WHITE, rect.center, 3)

    # Pacman
    pr = player_pos[0] * TILE_SIZE + TILE_SIZE // 2
    pc = player_pos[1] * TILE_SIZE + TILE_SIZE // 2
    pygame.draw.circle(screen, YELLOW, (pc, pr), TILE_SIZE // 2 - 2)

    # Ghosts
    for g in ghosts:
        gr = g["pos"][0] * TILE_SIZE + TILE_SIZE // 2
        gc = g["pos"][1] * TILE_SIZE + TILE_SIZE // 2
        pygame.draw.circle(screen, RED, (gc, gr), TILE_SIZE // 2 - 2)

    # HUD
    hud = font_med.render(f"Score: {score}   Dots left: {len(all_dots())}", True, ORANGE)
    screen.blit(hud, (8, 8))
    pygame.display.flip()

def show_screen(title, subtitle):
    """Show a message screen and wait for R / Q."""
    while True:
        screen.fill(BLACK)
        t1 = font_big.render(title,    True, WHITE)
        t2 = font_med.render(subtitle, True, GREEN)
        t3 = font_med.render("R = restart   Q = quit", True, ORANGE)
        screen.blit(t1, (WIDTH//2 - t1.get_width()//2, HEIGHT//2 - 60))
        screen.blit(t2, (WIDTH//2 - t2.get_width()//2, HEIGHT//2))
        screen.blit(t3, (WIDTH//2 - t3.get_width()//2, HEIGHT//2 + 50))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r: return
                if e.key == pygame.K_q: pygame.quit(); sys.exit()

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
def main():
    while True:
        clock.tick(10)

        # ── save positions BEFORE moving ──────────────────────────
        old_player = player_pos[:]
        old_ghosts = [g["pos"][:] for g in ghosts]

        # ── move ghosts ───────────────────────────────────────────
        move_ghosts()
        new_ghosts = [g["pos"][:] for g in ghosts]

        # ── COLLISION: did ghost walk INTO player this tick? ──────
        if collision_check(old_player, old_player, old_ghosts, new_ghosts):
            show_screen("GAME OVER", f"Score: {score}")
            reset_game()
            continue

        # ── compute danger map for this tick ──────────────────────
        danger = predict_ghost_cells(PREDICT_STEPS)
        dist   = min_ghost_dist()

        # ── decide Pacman's next step ─────────────────────────────
        next_step = None

        # PRIORITY 1 — hard escape when ghost is 1-2 tiles away
        if dist <= 2:
            next_step = escape_move()

        # PRIORITY 2 — safe A* to best dot
        if next_step is None:
            for target in best_targets(danger, n=12):
                path = astar(tuple(player_pos), target, danger, use_penalty=True)
                if path:
                    next_step = path[0]
                    break

        # PRIORITY 3 — A* ignoring ghost penalty (guarantees progress)
        if next_step is None:
            for target in best_targets(danger, n=12):
                path = astar(tuple(player_pos), target, danger, use_penalty=False)
                if path:
                    next_step = path[0]
                    break

        # PRIORITY 4 — random valid neighbor (last resort)
        if next_step is None:
            nb = neighbors(tuple(player_pos))
            if nb:
                next_step = random.choice(nb)

        # ── move Pacman ───────────────────────────────────────────
        if next_step:
            move_player(next_step)

        new_player = player_pos[:]

        # ── COLLISION: did Pacman walk into a ghost OR swap cells? ─
        if collision_check(old_player, new_player, old_ghosts, new_ghosts):
            show_screen("GAME OVER", f"Score: {score}")
            reset_game()
            continue

        # ── WIN check ─────────────────────────────────────────────
        if check_win():
            show_screen("YOU WIN! 🎉", f"Final Score: {score}")
            reset_game()
            continue

        # ── events ───────────────────────────────────────────────
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                pygame.quit(); sys.exit()

        draw()

main()
