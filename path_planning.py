import math
from typing import List, Tuple

Point = Tuple[int, int]


def calculate_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_closest_edge_point(point: Point, width: int, height: int) -> Tuple[float, Point]:
    x, y = point

    dists = [x, width - x, y, height - y]
    min_dist = min(dists)

    if min_dist == x:
        edge_point = (0, y)
    elif min_dist == width - x:
        edge_point = (width, y)
    elif min_dist == y:
        edge_point = (x, 0)
    else:
        edge_point = (x, height)

    return min_dist, edge_point


def plan_shortest_path(points: List[Point], width: int, height: int) -> Tuple[List[Point], float]:
    if not points:
        return [], 0.0

    best_total_path: List[Point] = []
    min_total_distance = float("inf")

    for start_node_idx in range(len(points)):
        first_point = points[start_node_idx]

        dist_to_wall, wall_start_point = get_closest_edge_point(first_point, width, height)

        current_path = [wall_start_point, first_point]
        current_distance = dist_to_wall
        unvisited = points[:start_node_idx] + points[start_node_idx + 1 :]
        current_pos = first_point

        while unvisited:
            nearest_idx = -1
            min_d = float("inf")

            for i, p in enumerate(unvisited):
                d = calculate_distance(current_pos, p)
                if d < min_d:
                    min_d = d
                    nearest_idx = i

            next_point = unvisited[nearest_idx]
            current_distance += min_d
            current_path.append(next_point)
            current_pos = next_point

            unvisited.pop(nearest_idx)

            if current_distance > min_total_distance:
                break

        if current_distance < min_total_distance and len(current_path) == len(points) + 1:
            min_total_distance = current_distance
            best_total_path = current_path

    return best_total_path, min_total_distance
