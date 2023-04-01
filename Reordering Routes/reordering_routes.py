from collections import defaultdict
def minReorder(n, connections):
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append((v, 1))
        graph[v].append((u, 0))
            
    def dfs(node):
        nonlocal total
        visited.add(node)
            
        for neighbor, cost in graph[node]:
            if neighbor not in visited:
                total += cost
                dfs(neighbor)
    total = 0
    visited = set()
    dfs(0)
    return total

n = 6
connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
print(minReorder(n, connections))