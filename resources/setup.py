import os
import urllib.request
import tarfile
import shutil


def bfs(graph: list, sourceNode: int, out_f: str):
    frontier, new_frontier = [sourceNode], []
    costs = [-1 for _ in range(len(graph))]
    costs[sourceNode] = 0
    cost = 0

    while frontier:
        cost += 1
        for node in frontier:
            for neighbour in graph[node]:
                if costs[neighbour] == -1:
                    costs[neighbour] = cost
                    new_frontier.append(neighbour)

        frontier, new_frontier = new_frontier, []

    print(f'bfs max costs for source node {sourceNode} = {cost}')

    with open(out_f, 'w') as f:
        f.write('\n'.join(map(str, costs)))

    return costs


def dijkstra(graph: list, wgraph: list, source: int, out_f: str):
    from heapq import heappop, heappush
    from _testcapi import FLT_MAX

    visited = [False] * len(graph)
    dist = [FLT_MAX for _ in range(len(graph))]
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

        for adjacent_node, adjacent_weight in zip(graph[node], wgraph[node]):
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


def to_adjacency_matrix(graph: list, wgraph: list, out_f: str):
    is_weighted = False
    for adj, w in zip(graph, wgraph):
        if 0 < len(adj):
            if len(adj) == len(w):
                is_weighted = True
            break

    with open(out_f, 'w') as f:
        # total edges
        if is_weighted:
            f.write('#weighted')
        else:
            f.write('#unweighted')
        f.write(os.linesep)

        f.write(str(len(graph)))
        f.write(os.linesep)
        for adj_list, weight_list in zip(graph, wgraph):
            f.write(str(len(adj_list)))
            f.write(os.linesep)
            if len(adj_list):
                f.write(' '.join(map(str, adj_list)))
                f.write(os.linesep)
                if weight_list:
                    f.write(' '.join(map(str, weight_list)))
                    f.write(os.linesep)

    print('Graph saved!')


def load_matrix_market(file_path: str, directed=False, weighted=False):
    graph = None
    wgraph = None
    with open(file_path, 'r') as f:
        dim = None
        i = 0
        for line in f:
            if line[0] == '%':
                continue
            elif dim is None:
                dim = list(map(int, line.strip().split(' ')))
                graph = [[] for _ in range(dim[0])]
                wgraph = [[] for _ in range(dim[0])]
            else:
                v1, v2, w = None, None, None
                if weighted:
                    l = line.strip().split(' ')
                    v2, v1 = map(int, l[:2])
                    w = float(l[2])
                else:
                    v2, v1 = map(int, line.strip().split(' '))

                v1, v2 = v1 - 1, v2 - 1

                graph[v1].append(v2)
                if weighted:
                    wgraph[v1].append(w)

                if not directed:
                    graph[v2].append(v1)
                    if weighted:
                        wgraph[v2].append(w)
                i += 1
                print(f'Loading graph: {int(i * 100 / dim[2])}%', end='\r')

    # for adj_list in graph:
    #     adj_list.sort()
    print('Graph loaded!')
    return graph, wgraph


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
        'appu.mtx': {
            'directed': True,
            'weighted': True,
            'link': 'http://www.cise.ufl.edu/research/sparse/MM/Simon/appu.tar.gz',
        },
        'ca2010.mtx': {
            'directed': False,
            'weighted': True,
            'link': 'https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/ca2010.tar.gz',
        },
        'cage13.mtx': {
            'directed': True,
            'weighted': True,
            'link': 'http://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage13.tar.gz',
        },
        # 'wiki-Talk.mtx': {
        #     'directed': True,
        #     'weighted': False,
        #     'link': '',
        # },
        'kron_g500-logn16.mtx': {
            'directed': False,
            'weighted': True,
            'link': 'http://www.cise.ufl.edu/research/sparse/MM/DIMACS10/kron_g500-logn16.tar.gz',
        },
    }
    for file, prop in files.items():
        print(f'> {file}')
        directed, weighted, link = prop['directed'], prop['weighted'], prop['link']
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

        graph, wgraph = load_matrix_market(file, directed=directed, weighted=weighted)
        to_adjacency_matrix(graph, wgraph, file + '.adj')
        for source_node in range(1):
            bfs(graph, source_node, file + '.ans')
            if weighted:
                dijkstra(graph, wgraph, source_node, file+'.ans.sssp')
        print()
