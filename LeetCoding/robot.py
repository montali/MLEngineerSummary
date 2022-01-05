cache = {}


def robot_walk(end, grid, start=[0, 0], past_path=[]):  # Add past path to avoid loops
    if start in past_path:
        return 0
    if start == end:
        return 1
    sum = 0
    if start[0] + 1 <= end[0] and not grid[start[0] + 1][start[1]]:
        sum += robot_walk(end, grid, [start[0] + 1, start[1]], past_path + [start])
    if start[1] + 1 <= end[0] and not grid[start[0]][start[1] + 1]:
        sum += robot_walk(end, grid, [start[0], start[1] + 1], past_path + [start])
    return sum


print(robot_walk([2, 2], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
