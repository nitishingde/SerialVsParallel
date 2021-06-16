import os
import urllib.request
import tarfile
import shutil


class Graph:
    def __init__(self, nodes: int, weighted: bool, directed: bool):
        self.dict = [[] for _ in range(nodes)]
        self.weighted = weighted
        self.directed = directed

    def add_edge(self, vertex1: int, vertex2: int, weight=0.0):
        self.dict[vertex1].append((vertex2, weight))
        if not self.directed and vertex1 != vertex2:
            self.dict[vertex2].append((vertex1, weight))

    def sort(self):
        for edgeList in self.dict:
            edgeList.sort()

    def __len__(self):
        return len(self.dict)


def bfs(graph: Graph, sourceNode: int, out_f: str):
    frontier, new_frontier = [sourceNode], []
    costs = [-1 for _ in range(len(graph))]
    costs[sourceNode] = 0
    cost = 0

    while frontier:
        cost += 1
        for node in frontier:
            for neighbour, _ in graph.dict[node]:
                if costs[neighbour] == -1:
                    costs[neighbour] = cost
                    new_frontier.append(neighbour)

        frontier, new_frontier = new_frontier, []

    print(f'bfs max costs for source node {sourceNode} = {cost}')

    with open(out_f, 'w') as f:
        f.write('\n'.join(map(str, costs)))

    return costs


def dijkstra(graph: Graph, source: int, out_f: str):
    from heapq import heappop, heappush

    visited = [False] * len(graph)
    dist = [float('inf') for _ in range(len(graph))]
    prev = [None] * len(graph)
    dist[source] = 0
    priority_queue = [(0, source)]

    while priority_queue:
        nodedata = heappop(priority_queue)
        node = nodedata[1]
        costs = nodedata[0]
        visited[node] = True

        # optimisation
        if dist[node] < costs:
            continue

        for adjacent_node, adjacent_weight in graph.dict[node]:
            if visited[adjacent_node]:
                continue

            new_dist = dist[node] + adjacent_weight
            if new_dist < dist[adjacent_node]:
                dist[adjacent_node] = new_dist
                prev[adjacent_node] = node
                heappush(priority_queue, (new_dist, adjacent_node))

        # optimization
        # if node == destination:
        # return dist, prev
    with open(out_f, 'w') as f:
        f.write('\n'.join(map(str, dist)))

    return dist, prev


def to_adjacency_matrix(graph: Graph, out_f: str):
    is_weighted = graph.weighted

    with open(out_f, 'w') as f:
        # total edges
        if is_weighted:
            f.write('#weighted')
        else:
            f.write('#unweighted')
        f.write(os.linesep)

        f.write(str(len(graph)))
        f.write(os.linesep)
        for adj_list in graph.dict:
            f.write(str(len(adj_list)))
            f.write(os.linesep)
            if len(adj_list):
                f.write(' '.join(map(str, [neighbour[0] for neighbour in adj_list])))
                f.write(os.linesep)
                if is_weighted:
                    f.write(' '.join(map(str, [neighbour[1] for neighbour in adj_list])))
                    f.write(os.linesep)

    print('Graph saved!')


def load_matrix_market(file_path: str):
    graph = None
    with open(file_path, 'r') as f:
        dim = None
        i = 0
        obj, form, field, symmetry = f.readline().strip().split(' ')[1:]
        weighted = False if field == 'pattern' else True
        directed = True if symmetry == 'general' else False
        for line in f:
            if line[0] == '%':
                if 'undirected' in line:
                    directed = False
                continue
            elif dim is None:
                dim = list(map(int, line.strip().split(' ')))
                graph = Graph(max(dim[0], dim[1]), weighted=weighted, directed=directed)
            else:
                v1, v2, w = None, None, None
                l = line.strip().split(' ')
                v2, v1 = map(int, l[:2])
                v1, v2 = v1 - 1, v2 - 1
                if weighted:
                    w = float(l[2])

                graph.add_edge(v1, v2, w)
                i += 1
                print(f'Loading graph: {int(i * 100 / dim[2])}%', end='\r')

    graph.sort()
    print('Graph loaded!')
    return graph


def load_adjacency_matrix(file_path: str):
    with open(file_path, 'r') as f:
        N = int(f.readline().strip())
        graph = []
        for i in range(N):
            if int(f.readline().strip()) == 0:
                graph.append([])
            else:
                graph.append(list(map(int, f.readline().strip().split(' '))))
            print(f'{int(100 * i / N)}%', end='\r')

        print('Graph loaded!')
        return graph


def progress(block_num, block_size, total_size):
    print(f"Downloading: {(100 * block_num * block_size) // total_size}%", end='\r')


def setup_primes():
    compressed_file = tarfile.open('primes1e6.tar.xz')
    compressed_file.extractall()
    compressed_file.close()


if __name__ == '__main__':
    setup_primes()

    files = {
        'appu.mtx': 'http://www.cise.ufl.edu/research/sparse/MM/Simon/appu.tar.gz',
        'ca2010.mtx': 'https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/ca2010.tar.gz',
        'cage13.mtx': 'http://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage13.tar.gz',
        'kron_g500-logn16.mtx': 'http://www.cise.ufl.edu/research/sparse/MM/DIMACS10/kron_g500-logn16.tar.gz',
        'gre_512.mtx': 'https://www.cise.ufl.edu/research/sparse/MM/HB/gre_512.tar.gz',
        'cage8.mtx': 'https://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage8.tar.gz',
    }
    for file, link in files.items():
        print(f'> {file}')
        compressed_file = link.split('/')[-1]
        if not os.path.isfile(file):
            if not os.path.isfile(compressed_file):
                print(f'Downloading {compressed_file}')
                urllib.request.urlretrieve(link, link.split('/')[-1], progress)
            # extract files
            print(f'Extracting {compressed_file}!')
            tar = tarfile.open(compressed_file)
            tar.extractall()
            tar.close()
            os.remove(compressed_file)
            for dirpath, dirnames, filenames in os.walk('./'):
                if file in filenames:
                    shutil.copy(os.path.join(dirpath, file), './')
                    shutil.rmtree(dirpath)
                    break

        if os.path.isfile(file+'.adj') and os.path.isfile(file+'.ans') and os.path.isfile(file+'.ans.sssp'):
            continue

        graph = load_matrix_market(file)
        if not os.path.isfile(file+'.adj'):
            to_adjacency_matrix(graph, file + '.adj')
        for source_node in range(1):
            if not os.path.isfile(file+'.ans'):
                bfs(graph, source_node, file + '.ans')
            if graph.weighted and not os.path.isfile(file+'.ans.sssp'):
                dijkstra(graph, source_node, file+'.ans.sssp')
        print()
