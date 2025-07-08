import time
import util
import networkx as nx
from itertools import islice
import numpy as np

# [수정]
def compute_rehandling_risk(k, node_allocations, K, N):
    origin, destination = K[k][0]
    quantity = K[k][1]

    risk = (destination - origin) * 1.0
    risk += quantity * 0.5

    congestion = 0
    for i in range(N):
        if node_allocations[i] != -1 and abs(i - destination) <= 3:
            congestion += 1
    risk += congestion * 0.3

    return risk

def algorithm(prob_info, timelimit=60):
    """
    This is a template for the custom algorithm.
    The algorithm should return a solution that is a list of (route, k) pairs.
    Each route is a list of node indices, and k is the index of the demand that is moved by the route.
    You CANNOT change or remove this function signature.
    But it is fine to define extra functions or mudules that are used in this function.
    """

    # [수정]
    def compute_total_risk(node_allocations, K, N):
        total = 0
        for i in range(N):
            k = node_allocations[i]
            if k != -1:
                total += compute_rehandling_risk(k, node_allocations, K, N)
        return total


    def generate_neighbors(node_allocations):
        neighbors = []
        N = len(node_allocations)
        for i in range(N):
            for j in range(i+1, N):
                if node_allocations[i] != -1 and node_allocations[j] != -1 and node_allocations[i] != node_allocations[j]:
                    new_alloc = node_allocations.copy()
                    new_alloc[i], new_alloc[j] = new_alloc[j], new_alloc[i]
                    neighbors.append(new_alloc)
        return neighbors

    # [수정]
    def delta_risk(current_alloc, i, j, current_score, K, N):
        """
        i와 j 노드를 swap했을 때 리스크 변화량만 계산해서 새로운 총 점수 반환
        """
        k_i = current_alloc[i]
        k_j = current_alloc[j]

        if k_i == -1 or k_j == -1 or k_i == k_j:
            return float('inf')  # 불필요한 스왑 무시

        # 원래 리스크 제거
        prev_risk_i = compute_rehandling_risk(k_i, current_alloc, K, N)
        prev_risk_j = compute_rehandling_risk(k_j, current_alloc, K, N)

        # 새로운 배치를 따로 생성
        temp_alloc = current_alloc.copy()
        temp_alloc[i], temp_alloc[j] = k_j, k_i

        # 새로운 리스크 계산
        new_risk_i = compute_rehandling_risk(k_j, current_alloc, K, N)
        new_risk_j = compute_rehandling_risk(k_i, current_alloc, K, N)

        delta = (new_risk_i + new_risk_j) - (prev_risk_i + prev_risk_j)
        return current_score + delta

    # [수정]
    def tabu_search(node_allocations, K, N, max_iter=30, tabu_size=5):
        best_alloc = node_allocations.copy()
        best_score = compute_total_risk(best_alloc, K, N)

        current = best_alloc.copy()
        current_score = best_score  # 점수 캐싱
        tabu_list = []
        no_improve = 0

        for _ in range(max_iter):
            neighbors = []

            for i in range(N):
                for j in range(i + 1, N):
                    if current[i] != -1 and current[j] != -1 and current[i] != current[j]:
                        neighbors.append((i, j))

            np.random.shuffle(neighbors)
            neighbors = neighbors[:100]  # 제한

            best_neighbor = None
            best_neighbor_score = float('inf')

            for i, j in neighbors:
                candidate = current.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]

                if any(np.array_equal(candidate, t) for t in tabu_list):
                    continue

                score = delta_risk(current, i, j, current_score, K, N)

                if score < best_neighbor_score:
                    best_neighbor = candidate
                    best_neighbor_score = score
                    best_i, best_j = i, j

            if best_neighbor is None:
                break

            current = best_neighbor
            current_score = best_neighbor_score
            tabu_list.append(current.copy())

            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            if current_score < best_score:
                best_alloc = current.copy()
                best_score = current_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 10:
                    break

        return best_alloc


    #------------- begin of custom algorithm code --------------#

    # This is an example of a custom algorithm.
    # It is a very simple greedy algorithm that loads and unloads the demands at each port.
    # It is not a very good algorithm, but it is a starting point for you to build your own algorithm.
    
    # A brief description of the algorithm:
    # This algorithm uses two heuristics:
    # 1. Loading heuristic: Load the demands at the port in a greedy way.
    # 2. Unloading heuristic: Unload the demands at the port in a greedy way.
    # By "greedy way", we mean that we try to load/unload the demands in the order of their distance from the gate.
    # For example, we try to load the demands destined to the later ports in the farthest reachable nodes from the gate.
    # If there are not enough reachable nodes, we rehandle the demands by rolling off the demands that are blocking the path to available nodes.
    # When unloading, if there is no exiting path to the gate, we unload the blocking demands first and then unload the demand at the port.
    # To address blocking demands, we pre-generate k-shortest paths from the gate to all nodes, which are considered to find the least blocking path to the gate.




    # Loading heuristic
    def loading_heuristic(p, node_allocations, rehandling_demands, K, N):

        print(f'Starting loading phase...')

        # All demands that should be loaded at port p
        K_load = {idx: r for idx, ((o,d),r) in enumerate(K) if o == p}

        # [수정]
        print(f"[DEBUG] K_load at port {p} = {K_load}")
        print(f"[DEBUG] type(K_load) = {type(K_load)}")

        if K_load is None:
            raise ValueError(f"K_load is None at port {p} — this must not happen")

        if not isinstance(K_load, dict):
            raise TypeError(f"K_load must be dict but got {type(K_load)}")

        if len(K_load) == 0 and len(rehandling_demands) == 0:
            print(f"[INFO] No demands to load at port {p}. Skipping loading heuristic.")
            return [], node_allocations

        print(f'Demands that are originated from this port: {K_load}')

        if len(rehandling_demands) > 0:
            print(f'Rehandled demands from unloading phase: {rehandling_demands}')
            # Merge the rehandling demands with the loading demands
            for k in rehandling_demands:
                if k in K_load:
                    K_load[k] += 1
                else:
                    K_load[k] = 1


        route_list = []

        last_rehandling_demands = []


        # Total number of demands to load (including rehandling demands)
        total_loading_demands = sum([r for k,r in K_load.items()])

        # Get reachable nodes from the gate
        reachable_nodes, reachable_node_distances = util.bfs(G, node_allocations)

        # Get not occupied nodes
        available_nodes = util.get_available_nodes(node_allocations)

        if len(available_nodes) < total_loading_demands:
            print(f"Not enough available nodes ({len(available_nodes)} are available) to load {total_loading_demands} at port {p}.")
            raise Exception("Not enough available nodes to load demand!!! This must not happen!!!")

        if len(reachable_nodes) < total_loading_demands:
            print(f"Not enough reachable nodes ({len(reachable_nodes)} are reachable) to load {total_loading_demands} demands at port {p}. Rehandling...")

            # A very simple rehandling heuristic
            # 1. Get nodes that are available but not reachable
            # 2. Loop until we have enough reachable nodes
            # 2-1. Pick a node from the available but not reachable nodes
            # 2-2. Get the shortest path to the node from the gate
            # 2-3. Roll-off demands occupied on the path by order of distance from the gate. (and push to rehandling_demands stack for the later reloading)
            # 2-4. Check if the number of reachable nodes is enough to load the demand
            available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]

            while len(reachable_nodes) < total_loading_demands:
                
                if len(available_but_not_reachable) == 0:
                    print("Not enough available_but_not_reachable nodes to rehandle. This must not happen!!!")
                    raise Exception("Not enough available_but_not_reachable nodes to rehandle. This must not happen!!!")
                
                # Pick a node from the available but not reachable nodes
                n = available_but_not_reachable.pop(0)

                # Get the shortest path to the node from the gate
                distances, previous_nodes = util.dijkstra(G, node_allocations=None)

                path = util.path_backtracking(previous_nodes, 0, n)

                # Roll-off blocking demands on the path by order of distance from the gate. (and push to last_rehandling_demands list for the later reloading)
                for idx, i in enumerate(path[:-1]):
                    if node_allocations[i] != -1:
                        # Node i is occupied by demand node_allocations[i]
                        last_rehandling_demands.append(node_allocations[i])
                        node_allocations[i] = -1

                        # Rehandling route (from the blocking node to the gate)
                        route_list.append((path[:idx+1][::-1], node_allocations[i]))

                        # Increasing the number demands to load
                        total_loading_demands += 1

                # Check if the number of reachable nodes is enough to load the demand
                reachable_nodes, reachable_node_distances = util.bfs(G, node_allocations)

            print(f'After rehandling, we have {len(reachable_nodes)} reachable nodes now! (but have to rehandle {len(last_rehandling_demands)} demands)')


        # Merge the rehandling demands with the loading demands
        for k in last_rehandling_demands:
            if k in K_load:
                K_load[k] += 1
            else:
                K_load[k] = 1

        print(f'total_loading_demands = {total_loading_demands}')

        if total_loading_demands > 0:

            # We take the fartest nodes from the gate to load the demands
            loading_nodes = reachable_nodes[-total_loading_demands:][::-1]

            # Sort the loading demands by destination ports
            # [수정 전] 도착 포트 기준 정렬 (baseline 전략)
            # sorted_K_load = sorted(K_load.items(), key=lambda x: K[x[0]][0][1], reverse=True)

            # [수정 후] Lookahead Risk 기반 정렬
            sorted_K_load = sorted(
                K_load.items(),
                key=lambda x: compute_rehandling_risk(x[0], node_allocations, K, N)
            )
                        
            # Flatten the loading demands
            flattened_K_load = []
            for k, r in sorted_K_load:
                for _ in range(r):
                    flattened_K_load.append(k)

            assert(len(flattened_K_load) == len(loading_nodes))

            # Get the shortest path to the node from the gate
            distances, previous_nodes = util.dijkstra(G, node_allocations)

            # Allocate the nodes starting from behind so that there is no blocking
            for n, k in zip(loading_nodes, flattened_K_load):
                node_allocations[n] = k

                path = util.path_backtracking(previous_nodes, 0, n)

                route_list.append((path, k))

        print(f'Loading phase completed with {len(route_list)} routes including {len(last_rehandling_demands)} rehandling routes!')

        return route_list, node_allocations


    # Unloading heuristic
    def unloading_heuristic(p, node_allocations):

        # All demands that should be unloaded at port p
        K_unload = {idx: r for idx, ((o,d),r) in enumerate(K) if d == p}

        print(f'Starting unloading phase...')
        print(f'Demands that are destined to this port: {K_unload}')

        route_list = []

        rehandling_demands = []

        for k, r in K_unload.items():
            unloading_nodes = sorted([n for n in range(N) if node_allocations[n] == k])

            # if r != len(unloading_nodes):
            #     print(f"==> Demand {k} at port {p} is not loaded correctly. Demand {k} should be {r} but it is {len(unloading_nodes)}")

            for n in unloading_nodes:

                if node_allocations[n] == -1:
                    # The node may be cleared already for rehandling... we skip if it is the case.
                    continue

                num_blocking_nodes = []

                # To unload a demand, we consider k-shortest paths to the gate node
                for path in shortest_paths[n]:
                    # For each path, we check if there are blocking nodes
                    blocking_nodes = [i for i in path[:-1] if node_allocations[i] != -1]
                    num_blocking_nodes.append((len(blocking_nodes), blocking_nodes, path))
                    
                # The path with the minimum blocking nodes
                min_blocking_nodes = sorted(num_blocking_nodes, key=lambda x: x[0])[0]

                if min_blocking_nodes[0] == 0:
                    # No blocking nodes exist
                    path = min_blocking_nodes[2]
                    route_list.append((path[::-1], k))
                    node_allocations[n] = -1
                else:
                    # We first unload the blocking nodes
                    blocking_nodes = min_blocking_nodes[1]
                    path = min_blocking_nodes[2]
                    num_blocking = 0
                    for bn in blocking_nodes:
                        
                        # Trim the path after the blocking node
                        unload_route = []
                        for i in path:
                            unload_route.append(i)
                            if bn == i:
                                break
                        route_list.append((unload_route[::-1], node_allocations[bn]))


                        # Check if the blocking demand is in the unloading list
                        if node_allocations[bn] not in K_unload:
                            # Rehandle the demand
                            rehandling_demands.append(node_allocations[bn])
                            num_blocking += 1
                            
                        # Roll-off the demand
                        node_allocations[bn] = -1
                        
                    route_list.append((path[::-1], k))
                    node_allocations[n] = -1
                    
                    if num_blocking>0: 
                        print(f"Unloading path for demand {k} at node {n} is blocked by {num_blocking} nodes. Rehandling...")

        print(f'Unloading phase completed with {len(route_list)} routes including {len(rehandling_demands)} rehandling routes!')

        return route_list, rehandling_demands



    # Get parameters from the prob_info dictionary
    N = prob_info['N']
    E = prob_info['E']
    E = set([(u,v) for (u,v) in E])
    K = prob_info['K']
    P = prob_info['P']
    F = prob_info['F']

    start_time = time.time()

    # Create a graph for the problem
    G = nx.Graph()

    G.add_nodes_from(range(N))

    G.add_edges_from(E)

    # Shortest distances from the gate
    shortest_distances = np.zeros(N, dtype=int)


    # Generate k-shortest paths from the gate to all nodes
    max_num_paths = 5

    shortest_paths = [[]] 

    for i in range(1, N):

        sp_i = list(islice(nx.shortest_simple_paths(G, 0, i), max_num_paths))

        shortest_paths.append(sp_i)

        shortest_distances[i] = len(sp_i[0]) - 1


    # Convert the shortest paths to a numpy array using indicator vectors
    shortest_paths_array = [[]] 

    for i in range(1, N):
        sp_i_array = []
        for p in shortest_paths[i]:
            path_array = np.zeros(N, dtype=int)
            path_array[p[:-1]] = 1

            sp_i_array.append(path_array)
                
        shortest_paths_array.append(sp_i_array)




    # Current status of nodes
    # -1 means available (i.e. empty)
    # otherwise means occupied with a demand
    node_allocations = np.ones(N, dtype=int) * -1


    # Solution dictionary
    solution = {
        p: []
        for p in range(P)
    }


    # Loop over all ports and apply the loading and unloading heuristics to generate the routes
    for p in range(P):

        print(f"Port {p} ==============================")

        if p > 0:
            # Unloading heuristic
            route_list_unload, rehandling_demands = unloading_heuristic(p, node_allocations)

            solution[p].extend(route_list_unload)
        else:
            rehandling_demands = []

        # [수정 전]
        # if p < P-1:
        #    # Loading heuristic
        #    route_list_load, node_allocations = loading_heuristic(p, node_allocations, rehandling_demands)
        #
        #    solution[p].extend(route_list_load)

        #[수정]
        if p < P-1:
            # Step 1: 초기 greedy 할당 (route는 무시)
            route_list_load, node_allocations = loading_heuristic(p, node_allocations, rehandling_demands, K, N)

            # Step 2: Tabu Search로 할당 최적화
            node_allocations = tabu_search(node_allocations, K, N)

            # Step 3: 최적화된 할당 기반 경로 재생성
            distances, previous_nodes = util.dijkstra(G, node_allocations=None)
            route_list_refined = []
            for i in range(N):
                k = node_allocations[i]
                if k != -1:
                    path = util.path_backtracking(previous_nodes, 0, i)
                    route_list_refined.append((path, k))

            # Step 4: 재생성된 route 저장
            solution[p].extend(route_list_refined)
        
        print()


    #------------- end of custom algorithm code --------------#



    return solution



if __name__ == "__main__":
    # You can run this file to test your algorithm from terminal.

    import json
    import os
    import sys
    import jsbeautifier


    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        raise TypeError(f"Type {type(obj)} not serializable")
    
    
    # Arguments list should be problem_name, problem_file, timelimit (in seconds)
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)

        exception = None
        solution = None

        try:

            alg_start_time = time.time()

            # Run algorithm!
            solution = algorithm(prob_info, timelimit)

            alg_end_time = time.time()


            checked_solution = util.check_feasibility(prob_info, solution)

            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2 # allowing additional 2 second!
            checked_solution['exception'] = exception

            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file


            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')
                
            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)

