import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from numpy.linalg import norm
import networkx as nx
from shapely import MultiPoint, Point
from python_tsp.exact import solve_tsp_dynamic_programming
from json import dumps

MAX_TOTAL_WAYLENGTH = 360  # min., min = 60; = day length
MAX_SINGLE_TIME_DISTANCE = 120  # min.
MAX_CORE_GROUP_COVERAGE = 0.75
MAX_STATIONS_ADAY_TO_OMIT_GROUP_COVERAGE = 4
MAX_ON_THE_WAY_EXTENSION = 1.75  # relative factor
# min. to the core group, should be less than MAX_SINGLE_TIME_DISTANCE
BLOCK_ON_THE_WAY_ABOVE = 90
ON_THE_WAY_ANGLE_CRITERION = 45  # degrees (both sides of the line)
OPTIMIZE_REMOVE_STATIONS_UNTIL = 2  # remaining at one day
# very very slow but usually more accurate (uses TSP on full graph)
SLOW_ALGORITHM = False
EARTH_R = 6371000  # m
LEAF_SIZE = 40
# keys: idx (int), values: home dicts with keys: 'name', 'lat', 'lon', 'stay' (int, days)
homes_dict = []
# keys: 'name', 'lat', 'lon', 'duration' (int, minutes -> idle time), 'favorite' (0/1)
unclass_poi_dict = {}
selected_home = 0  # index of selected home
# keys: idx (int) (same as homes_dict -> those poi belong to that home; association is json_payload for external("cache_poi_assoc") ), values: poi dicts with keys: 'name', 'lat', 'lon', 'duration' (int, minutes, -> idle time), 'favorite' (0/1)
poi_dict = [][selected_home]


async def external(service, json_payload):
    # IMPLEMENT YOURSELF
    # if service == "route":
    #    json_payload: {"start": {"lon": float, "lat": float}, "end": {"lon": float, "lat": float}}
    #    ... send to external function
    #    return: int (time in minutes)
    # elif service == "cache_poi_assoc":
    #    json_payload: List of dicts with 'lat', 'lon', 'lat_center', 'lon_center'
    #    ... send to external function
    #    return: int (number of cached entries)
    pass


print("ROUTING: Commencing Task")

# CONTRIBUTED FUNCTIONS

# Python program to left-rotate the given array
# https://www.geeksforgeeks.org/python/python-program-for-program-for-array-rotation-2/
# This code was contributed by SR.Dhanush

# Function reverse the given array
# by swapping first and last numbers.


def reverse(start, end, arr):

    # No of iterations needed for reversing the list
    no_of_reverse = end-start+1

    # By incrementing count value swapping
    # of first and last elements is done.
    count = 0
    while ((no_of_reverse)//2 != count):
        arr[start+count], arr[end-count] = arr[end-count], arr[start+count]
        count += 1
    return arr

# Function takes array, length of
# array and no of rotations as input


def left_rotate_array(arr, size, d):

    # Reverse the Entire List
    start = 0
    end = size-1
    arr = reverse(start, end, arr)

    # Divide array into twosub-array
    # based on no of rotations.
    # Divide First sub-array
    # Reverse the First sub-array
    start = 0
    end = size-d-1
    arr = reverse(start, end, arr)

    # Divide Second sub-array
    # Reverse the Second sub-array
    start = size-d
    end = size-1
    arr = reverse(start, end, arr)
    return arr

# END CONTRIBUTED FUNCTIONS


meters = np.vectorize(lambda deg: deg * EARTH_R * np.pi / 180)


def geometry(df):
    return list(df.apply(lambda geom: [geom["lon"], geom["lat"]], axis=1))


def df_distances(df_from, df_to, k_neighbor=1):
    right = np.vstack(geometry(df_to))
    left = np.vstack(geometry(df_from))
    tree = BallTree(right, leaf_size=LEAF_SIZE)
    return tree.query(left, k_neighbor)


def sp_distance(p_from, p_to):
    return meters(Point([p_from.lon, p_from.lat]).distance(Point([p_to.lon, p_to.lat])))


def build_network(center, stations, max_neighbors=5):
    assert center["stay"] > 0
    assert stations.shape[0] > 0
    G = nx.Graph()

    # get neighbors
    dist, indices = df_distances(stations, stations, min(
        max_neighbors+1, len(stations)))  # +1 to include self

    G.add_nodes_from(indices[:, 0])
    for neighbor_k in range(max_neighbors):
        for source_station in indices[:, 0]:
            target_station = indices[source_station,
                                     neighbor_k + 1]  # +1 to skip self
            if not G.has_edge(source_station, target_station):
                # add edge with distance as weight
                G.add_edge(source_station, target_station,
                           weight=dist[source_station, neighbor_k + 1])

        # check if graph meets requirements
        if nx.number_connected_components(G) <= center["stay"]:
            break
        elif neighbor_k == max_neighbors - 1:
            return build_network(center, stations, max_neighbors * 2)

    # rename nodes to station names
    G = nx.relabel_nodes(G, {i: stations.at[i, "name"] for i in G.nodes()})
    return G


def get_core_groups(G, center, stations):
    core_groups = []
    removed_edges = []
    omit_coverage_check = G.number_of_nodes(
    ) / center["stay"] <= MAX_STATIONS_ADAY_TO_OMIT_GROUP_COVERAGE
    while True:
        # remove longest edge if needed
        if (not omit_coverage_check) or nx.number_connected_components(G) < center["stay"]:
            longest = max(G.edges.data(
                "weight"), key=lambda x: x[2])
            removed_edges.append(longest)
            G.remove_edge(*longest[:2])
        if nx.number_of_edges(G) == 0:
            break

        # determine n core groups (including favorite nodes or most stations), n = duration of stay
        favorite_groups = [frozenset(nx.node_connected_component(
            G, station)) for station in stations[stations["favorite"] == 1]["name"]]
        favorite_groups = list(
            map(set, list(set(favorite_groups))))  # remove duplicates
        assert center["stay"] >= len(
            favorite_groups), "Not enough duration to cover favorite groups"
        if center["stay"] == len(favorite_groups):
            core_groups = favorite_groups
        else:
            # sort by number of stations first, path length estimate (perimeter + smallest distance to center) second
            core_groups = list(sorted(nx.connected_components(G),
                                      key=lambda n: (len(n),
                                                     1/((0 if len(n) <= 2 else 2*np.sqrt(np.pi*MultiPoint(geometry(stations[stations["name"].isin(n)])).convex_hull.area))
                                                     + min(list(stations[stations["name"].isin(n)]["distance"])))
                                                     ),
                                      reverse=True))
            core_groups = list(
                filter(lambda x: x not in favorite_groups, core_groups))
            core_groups = core_groups[:center["stay"]-len(favorite_groups)]
            core_groups.extend(favorite_groups)

        if omit_coverage_check:
            if len(core_groups) == center["stay"]:
                break
        else:
            # check if max xx% of stations are in core groups
            nodes_in_core = sum(len(group) for group in core_groups)
            if nodes_in_core <= MAX_CORE_GROUP_COVERAGE * G.number_of_nodes():
                break
    return core_groups, G, removed_edges


async def get_route(start, end):
    api_time = await external("route", dumps({"start": {"lon": float(start.lon), "lat": float(start.lat)}, "end": {"lon": float(end.lon), "lat": float(end.lat)}}))
    assert api_time > 0
    return api_time


async def fetch_additional_routes(ref_group, core_group, stations):
    # get 3 geographically closest pairs of points
    r_stations = stations[stations["name"].isin(
        ref_group)].reset_index(drop=True)
    c_stations = stations[stations["name"].isin(
        core_group)].reset_index(drop=True)
    pairs = pd.merge(r_stations, c_stations,
                     how="cross", suffixes=("_r", "_c"))
    pairs["distance"] = pairs.apply(
        lambda row: sp_distance(row[["lon_r", "lat_r"]].rename({"lon_r": "lon", "lat_r": "lat"}),
                                row[["lon_c", "lat_c"]].rename({"lon_c": "lon", "lat_c": "lat"})), axis=1
    )
    pairs = [[p["name_r"], p["name_c"], p["distance"]]
             for _, p in pairs.iterrows()]
    pairs = list(sorted(pairs, key=lambda p: p[2]))[:3]
    for p in range(len(pairs)):
        start = stations[stations["name"]
                         == pairs[p][0]].iloc[0]
        end = stations[stations["name"]
                       == pairs[p][1]].iloc[0]
        edge_time = await get_route(start, end)
        pairs[p][2] = edge_time
        pairs[p] = tuple(pairs[p])
    return pairs


async def associate_remaining_groups(remaining, core_groups, removed_edges, stations, center):
    '''
    RETURNS
    -------
    A dictionary. Keys: core group indices (int) as in the given array.
    Values: List of Tuples each with associated remaining group (Pos 1)
    and relative minimum extra time to reach core group if included (Pos 2)
    '''
    # associate remaining components/groups to core groups if they are 'on the way' to the core group
    smallest_ways_core = list(map(lambda g: stations.loc[stations[stations["name"].isin(g)][
                              "distance"].idxmin()], core_groups))
    smallest_ways_core_stations = [swc["name"] for swc in smallest_ways_core]
    smallest_ways_core = [swc["distance"] for swc in smallest_ways_core]

    on_the_way = {i: [] for i in range(len(core_groups))}
    ways_to_all_cores = {i: [] for i in range(len(core_groups))}
    for r in remaining:
        smallest_way = stations.loc[stations[stations["name"].isin(
            r)]["distance"].idxmin()]
        smallest_way_station = smallest_way["name"]
        smallest_way = smallest_way["distance"]

        # get distances for smallest connection from home to start point of core group if on-the-way candidate was included
        ways_to_core = []
        for c in core_groups:
            # check if edge to core group existed before (on-the-way candidate)
            remaining_to_core = list(sorted(filter(lambda e: (e[0] in r and e[1] in c) or (
                e[0] in c and e[1] in r), removed_edges), key=lambda e: e[2]))
            if len(remaining_to_core) > 0:
                ways_to_core.append(smallest_way + remaining_to_core[0][2])
                ways_to_all_cores[core_groups.index(c)].append(
                    (r, remaining_to_core[0][2], 0))
            else:
                r_hull = MultiPoint(geometry(stations[stations["name"].isin(
                    r)])).convex_hull
                c_hull = MultiPoint(geometry(stations[stations["name"].isin(
                    c)])).convex_hull
                geographic_distance = meters(r_hull.distance(c_hull))
                ways_to_all_cores[core_groups.index(c)].append(
                    (r, geographic_distance, 1))
                # check if candidate could be on the way by: distance (smaller than core group) and angle (< xx° to core group)
                # no on-the-way check for stations over xx m
                if smallest_ways_core[core_groups.index(c)] <= BLOCK_ON_THE_WAY_ABOVE:
                    # get angle to core group
                    origin = np.array([center.lon, center.lat])
                    point_a = stations[stations["name"] ==
                                       smallest_way_station].iloc[0]
                    vector_a = np.array([point_a.lon, point_a.lat]) - origin
                    point_b = stations[stations["name"] ==
                                       smallest_ways_core_stations[core_groups.index(c)]].iloc[0]
                    vector_b = np.array([point_b.lon, point_b.lat]) - origin
                    angle = vector_a.dot(vector_b) / \
                        (norm(vector_a) * norm(vector_b))
                    if smallest_way < smallest_ways_core[core_groups.index(c)] and angle > np.cos(np.deg2rad(ON_THE_WAY_ANGLE_CRITERION)):
                        pairs = await fetch_additional_routes(r, c, stations)
                        removed_edges.extend(pairs)
                        ways_to_core.append(smallest_way + pairs[0][2])
                    else:
                        ways_to_core.append(0)
                else:
                    ways_to_core.append(0)

        # determine relative factor of extension caused by stop-by
        for i in range(len(core_groups)):
            ways_to_core[i] = (ways_to_core[i] / smallest_ways_core[i], i)
        # remove non-candidates
        ways_to_core = filter(lambda w: w[0] > 0, ways_to_core)
        # select the closest core group (for the on-the-way candidate) and check if it prolong the way only minimally
        ways_to_core = list(sorted(ways_to_core, key=lambda w: w[0]))
        if len(ways_to_core) > 0:
            if ways_to_core[0][0] <= MAX_ON_THE_WAY_EXTENSION:
                on_the_way[ways_to_core[0][1]].append((r, ways_to_core[0][0]))
    ways_to_all_cores = {k: [v[0] for v in sorted(
        ways_to_all_cores[k], key=lambda x: (x[2], x[1]))] for k in ways_to_all_cores}
    return on_the_way, removed_edges, ways_to_all_cores


# greedy DFS search; guarantees to return node counts/multiplication needed for a Hamiltonian path
def find_diversions(graph):
    visits = {n: 0 for n in list(graph)}  # returned
    backtrace = []
    nextNode = list(graph)[0]
    forwards = True
    while True:
        visits[nextNode] += 1
        if forwards:
            backtrace.append(nextNode)
        neighbors = np.array([[n, graph[nextNode][n]["weight"]]
                             for n in graph.neighbors(nextNode) if visits[n] == 0])
        if neighbors.size == 0:
            del backtrace[-1]
            if np.all(np.array(list(visits.values())) > 0) or len(backtrace) == 0:
                return visits
            forwards = False
            nextNode = backtrace[-1]
        else:
            forwards = True
            nextNode = str(neighbors[np.argmin(neighbors, axis=0)[1], 0])


async def optimize_route(day_graph, include_nodes, center_name, poi):
    include_nodes = include_nodes[:]
    # get distance matrix
    distance_matrix = nx.to_numpy_array(
        day_graph, nodelist=include_nodes, dtype=np.float16)
    distance_matrix[distance_matrix == 0] = np.inf
    np.fill_diagonal(distance_matrix, 0)

    # find optimal route with TSP
    permutation, optimal_distance = solve_tsp_dynamic_programming(
        distance_matrix)
    if optimal_distance != np.inf:
        optimal_route = [include_nodes[n] for n in permutation]
    else:
        day_graph = day_graph.copy()
        for node, occurences in find_diversions(day_graph).items():
            for _ in range(occurences-1):  # duplicate hubs
                include_nodes.append(node)
                ndx = include_nodes.index(node)  # only first index
                ndcol = distance_matrix[:, [ndx]]
                ndcol[ndcol == 0] = np.inf  # disconnect former self
                distance_matrix = np.hstack([distance_matrix, ndcol])
                ndrow = np.hstack([ndcol.T, [[0]]])
                distance_matrix = np.vstack([distance_matrix, ndrow])
        npermutation, _ = solve_tsp_dynamic_programming(
            distance_matrix)
        optimal_route = []  # clean route
        permutation = []
        optimal_distance = 0
        nname = None
        for n in npermutation:
            nname = include_nodes[n]
            if nname not in optimal_route:
                optimal_route.append(nname)
                permutation.append(n)
                if len(optimal_route) > 1:
                    nbf = optimal_route[-2]
                    if not day_graph.has_edge(nname, nbf):
                        edge = await get_route(
                            poi[poi["name"] == nname].iloc[0], poi[poi["name"] == nbf].iloc[0])
                        day_graph.add_edge(nname, nbf, weight=edge)
                        optimal_distance += edge
                    else:
                        optimal_distance += day_graph[nname][nbf]["weight"]
        nbf = optimal_route[0]
        if not day_graph.has_edge(nname, nbf):
            edge = await get_route(poi[poi["name"] == nname].iloc[0],
                                   poi[poi["name"] == nbf].iloc[0])
            day_graph.add_edge(nname, nbf, weight=edge)
            optimal_distance += edge
        else:
            optimal_distance += day_graph[nname][nbf]["weight"]
    idle_time = np.sum([poi[poi["name"] == r]["duration"].values[0]
                        for r in include_nodes if not r.startswith(center_name)])
    return optimal_route, optimal_distance+idle_time, permutation, distance_matrix, day_graph


async def draft_daygraphs(G, core_groups, remaining_groups, removed_edges, on_the_way, stations, center_name):
    # draft day graphs
    day_graphs = []
    preliminary_distances = []
    for c in range(len(core_groups)):
        day_graph = G.subgraph(core_groups[c]).copy()

        if len(on_the_way[c]) > 0:
            best_candidate = min(
                on_the_way[c], key=lambda candidate: candidate[1])[0]
            best_candidate_edges = filter(lambda e: (e[0] in best_candidate and e[1] in core_groups[c]) or (
                e[0] in core_groups[c] and e[1] in best_candidate), removed_edges)
            day_graph.add_nodes_from(best_candidate)
            day_graph.add_weighted_edges_from(best_candidate_edges)
            remaining_groups.remove(best_candidate)
            core_groups[c] = core_groups[c].union(best_candidate)

        # add center node
        day_stations = stations[stations["name"].isin(day_graph.nodes())]
        day_edges = zip([center_name for _ in range(
            day_graph.number_of_nodes())], day_stations["name"], day_stations["distance"])
        day_graph.add_node(center_name)
        day_graph.add_weighted_edges_from(day_edges)

        _, optimal_distance, _, _, day_graph = await optimize_route(
            day_graph, list(day_graph.nodes()), center_name, stations)
        day_graphs.append(day_graph)
        preliminary_distances.append(optimal_distance)
    return day_graphs, preliminary_distances, core_groups, remaining_groups


async def merge_associated(G, preliminary_distances, assoc_groups, remaining_groups, core_groups, removed_edges, stations, day_graphs, center_name):
    # merge small core groups with assoc groups
    min_route_index = np.argmin(preliminary_distances)
    while preliminary_distances[min_route_index] < (MAX_TOTAL_WAYLENGTH - 60):
        # get next available assoc group
        assoc = None
        while assoc not in remaining_groups:
            if len(assoc_groups[min_route_index]) == 0:
                assoc = None
                break
            assoc = assoc_groups[min_route_index][0]
            del assoc_groups[min_route_index][0]
        if assoc is None:
            preliminary_distances[min_route_index] = np.inf
            min_route_index = np.argmin(preliminary_distances)
            continue

        # add to graph and recalculate distance
        assoc_edges = list(filter(lambda e: (e[0] in assoc and e[1] in core_groups[min_route_index]) or (
            e[0] in core_groups[min_route_index] and e[1] in assoc), removed_edges))
        internal_edges = G.subgraph(assoc).edges.data("weight")
        edges_to_center = zip([center_name for _ in range(
            len(assoc))], assoc, stations[stations["name"].isin(assoc)]["distance"])
        if len(assoc_edges) < 2:
            assoc_edges = await fetch_additional_routes(
                assoc, core_groups[min_route_index], stations)
        day_graphs[min_route_index].add_nodes_from(assoc)
        day_graphs[min_route_index].add_weighted_edges_from(assoc_edges)
        day_graphs[min_route_index].add_weighted_edges_from(internal_edges)
        day_graphs[min_route_index].add_weighted_edges_from(edges_to_center)
        _, optimal_distance, _, _, day_graph = await optimize_route(
            day_graphs[min_route_index], list(day_graphs[min_route_index].nodes()), center_name, stations)
        day_graphs[min_route_index] = day_graph
        preliminary_distances[min_route_index] = optimal_distance
        remaining_groups.remove(assoc)
        core_groups[min_route_index] = core_groups[min_route_index].union(
            assoc)

        min_route_index = np.argmin(preliminary_distances)
    return core_groups, remaining_groups, day_graphs

# MAIN


async def main():
    homes = pd.DataFrame.from_records(homes_dict)
    center = homes.iloc[selected_home]
    poi = pd.DataFrame.from_records(unclass_poi_dict)
    stations = pd.DataFrame.from_records(poi_dict)

    if not poi.empty:
        # group by nearest home
        print("ROUTING: Grouping POI by nearest home")
        dist, indices = df_distances(poi, homes)
        dist = dist.flatten()
        indices = indices.flatten()

        poi["center"] = homes.iloc[indices]["name"].values
        poi["distance"] = meters(dist)
        assert not (poi["distance"] == 0).any()
        this_poi = poi[poi["center"] == center["name"]]
        if stations.empty:
            stations = this_poi.reset_index(drop=True)
        else:
            stations = pd.concat([stations, this_poi]).reset_index(drop=True)
        return_poi = poi[["lat", "lon", "center"]]
        return_poi = return_poi.join(homes.set_index(
            "name"), on="center", rsuffix="_center")
        return_poi = return_poi[[
            "lat", "lon", "lat_center", "lon_center"]].reset_index(drop=True)
        success = (await external("cache_poi_assoc", dumps(return_poi.to_dict("records")))).toPy()
        assert success > 0

    # remove stations with >= xx minute distance using API if not favorite
    print("ROUTING: Building network")
    for row in range(len(stations)):
        stations.loc[row, "distance"] = await get_route(center, stations.loc[row])
    stations = stations[(stations["favorite"] == 1) | (
        stations["distance"] <= MAX_SINGLE_TIME_DISTANCE)].reset_index(drop=True)

    G = build_network(center, stations)

    # update Graph weights using external function
    for edge in G.edges:
        start = stations[stations["name"] == edge[0]].iloc[0]
        end = stations[stations["name"] == edge[1]].iloc[0]
        edge_time = await get_route(start, end)
        if edge_time > MAX_SINGLE_TIME_DISTANCE:
            G.remove_edge(*edge[:2])
        else:
            G[edge[0]][edge[1]]["weight"] = edge_time

    if SLOW_ALGORITHM:
        print("ROUTING: drafting daygraphs. Please be patient")
        # add center
        G.add_node(center["name"])
        for _, station in stations.iterrows():
            G.add_edge(center["name"],
                       station["name"], weight=station["distance"])
        # add center copies
        for i in range(center["duration"]-1):
            G.add_node(f"{center['name']}_{i}")
            for _, station in stations.iterrows():
                G.add_edge(f"{center['name']}_{i}",
                           station["name"], weight=station["distance"])

        # optimize to get day graphs
        optimal_route, _, _, _, G = await optimize_route(
            G, list(G), center["name"], stations)
        print(optimal_route)

        day_graph_nodes = [[] for _ in range(center["duration"])]
        ix = -1
        lx = 0
        for rx in range(len(optimal_route)):
            if optimal_route[rx].startswith(center["name"]):
                day_graph_nodes[ix].append(center["name"])
                day_graph_nodes[ix].extend(optimal_route[lx:rx])
                lx = rx+1
                ix += 1
        day_graph_nodes[-1].extend(optimal_route[lx:])
        print(day_graph_nodes)

        day_graphs = [G.subgraph(nodes) for nodes in day_graph_nodes]
    else:
        print("ROUTING: Determining core groups")
        core_groups, G, removed_edges = get_core_groups(G, center, stations)
        print("ROUTING core groups:", dumps([list(c) for c in core_groups]))

        print("ROUTING: Associating remaining groups")
        remaining_groups = list(filter(
            lambda x: x not in core_groups, nx.connected_components(G)))
        on_the_way, removed_edges, assoc_groups = await associate_remaining_groups(
            remaining_groups, core_groups, removed_edges, stations, center)

        print("ROUTING: Drafting day graphs")
        day_graphs, preliminary_distances, core_groups, remaining_groups = await draft_daygraphs(
            G, core_groups, remaining_groups, removed_edges, on_the_way, stations, center["name"])

        core_groups, remaining_groups, day_graphs = await merge_associated(
            G, preliminary_distances, assoc_groups, remaining_groups, core_groups, removed_edges, stations, day_graphs, center["name"])
        print("ROUTING final groups:", dumps([list(c) for c in core_groups]))

    print("ROUTING: Optimizing daily routes")
    optimal_routes = []
    for d in range(len(day_graphs)):
        print("ROUTING Day", d+1)
        day_graph = day_graphs[d]
        # find optimal route travelling salesman + remove stations so that it fits max day duration; redo route
        optimal_route = []
        optimal_distance = np.inf
        include_nodes = list(day_graph.nodes())
        while True:
            optimal_route, optimal_distance, permutation, distance_matrix, day_graph = await optimize_route(
                day_graph, include_nodes, center["name"], stations)

            # if route is too small or fits day, break
            if len(include_nodes) <= (OPTIMIZE_REMOVE_STATIONS_UNTIL+1) or optimal_distance <= MAX_TOTAL_WAYLENGTH:
                break

            # else: remove nodes that make it to long
            bulge = []
            for p in range(len(permutation)):
                p_before = p-1
                p_after = p+1 if p != len(permutation) - 1 else 0
                # don't remove center or favorite stations
                if optimal_route[p] != center["name"] and stations[stations["name"] == optimal_route[p]].favorite.reset_index(drop=True)[0] != 1:
                    bulge.append((optimal_route[p],
                                  distance_matrix[permutation[p_before]][permutation[p]] +
                                  distance_matrix[permutation[p]][permutation[p_after]]))
            if len(bulge) == 0:
                break  # impossible to optimize
            max_bulge = max(bulge, key=lambda b: b[1])[0]
            include_nodes.remove(max_bulge)

            # reconnect graph
            parts = list(nx.connected_components(
                day_graph.subgraph(include_nodes)))
            if len(parts) > 1:
                for i in range(len(parts)):
                    for other_part in parts[i+1:]:
                        day_graph.add_edges_from(
                            fetch_additional_routes(other_part, parts[i]))

        # rotate optimal route
        optimal_route = left_rotate_array(optimal_route, len(
            optimal_route), optimal_route.index(center["name"]))[1:]
        route_coordinates = [{"lon": float(stations.loc[stations["name"] == poi, "lon"].iloc[0]), "lat": float(
            stations.loc[stations["name"] == poi, "lat"].iloc[0])} for poi in optimal_route]
        optimal_routes.append(
            {"route": route_coordinates, "duration": optimal_distance})

    print("ROUTING: Completed")
    return dumps(optimal_routes)

main()

