from itertools import combinations, permutations
from typing import Tuple

import networkx as nx
import osmnx as ox
from folium.plugins import Geocoder
from geopandas import GeoSeries
from geopy import distance
from networkx import MultiDiGraph
from osmnx import geocoder


class OSMGraph:
    def __init__(
        self,
        distance_kms: int,
        address: str,
    ) -> None:
        self.distance_kms = distance_kms
        self.address = address

    def add_interest_points(
        self, graph: MultiDiGraph, delivery_points: GeoSeries
    ) -> "MultiDiGraph":

        node_id = 1
        for point in delivery_points:

            print(f"processing point: {point}")
            self.add_connected_node_to_graph(
                graph, node_id, {"goal": "delivery_point"}, point.x, point.y
            )

            node_id += 1
        return graph

    def add_connected_node_to_graph(
        self,
        graph: MultiDiGraph,
        node,
        node_goal: dict[str, str],
        node_x,
        node_y,
        color="red",
    ):

        nearest_node = ox.distance.nearest_nodes(graph, node_x, node_y)
        distance_found = distance.distance(
            (graph.nodes[nearest_node]["y"], graph.nodes[nearest_node]["x"]),
            (node_y, node_x),
        )
        print(f"distance from {nearest_node} for {node} is {distance_found}")
        graph.add_node(node, x=node_x, y=node_y, attr=node_goal, color=color)
        # Add bidirectional edge
        graph.add_edge(nearest_node, node, weight=distance_found.m)
        graph.add_edge(node, nearest_node, weight=distance_found.m)
        return graph

    def generate_fake_delivery_graph(self) -> Tuple[GeoSeries, MultiDiGraph, str]:
        address = "La Madeleine,Nord,France"
        center_lat, center_long = geocoder.geocode(query=address)
        print(f"Starting point is lat:{center_lat} long:{center_long}")
        graph = ox.graph_from_place(
            address, network_type="drive", simplify=False, clean_periphery=False
        )
        graph = ox.project_graph(graph)

        return graph, center_lat, center_long

    def complete_graph(self, graph, nodes_list):
        G = nx.empty_graph(len(nodes_list), graph)
        if len(nodes_list) > 1:
            if G.is_directed():
                edges = permutations(nodes_list, 2)
            else:
                edges = combinations(nodes_list, 2)
            G.add_edges_from(edges)
        return G

    def find_speed(self, row):
        fast = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "escape",
            "track",
        ]
        slow = ["tertiary", "residential", "tertiary_link", "living_street"]
        other = ["unclassified", "road", "service"]
        if row["highway"] in fast:
            return 90 / 3.6
        elif row["highway"] in slow:
            return 50 / 3.6
        elif row["highway"] in other:
            return 30 / 3.6
        else:
            return 20 / 3.6

    def generate_graph_from_address(self) -> Tuple[MultiDiGraph, str]:
        ox.config(log_console=True, use_cache=True)

        graph, center_lat, center_long = self.generate_fake_delivery_graph()
        graph = ox.consolidate_intersections(
            graph, dead_ends=False, reconnect_edges=True, tolerance=15
        )
        print(f"bus stop found used as fake deliveries: {len(graph.nodes)}")

        graph = ox.utils_graph.remove_isolated_nodes(graph)

        depot = ox.distance.nearest_nodes(graph, center_lat, center_long)

        nodes_to_explore, edges = ox.graph_to_gdfs(
            graph, edges=True, nodes=True, node_geometry=True
        )
        edges = edges.assign(speed=edges.apply(self.find_speed, axis=1))
        edges["weight"] = round(edges["length"] / edges["speed"])
        print("Depot location is: ", graph.nodes[depot]["x"], graph.nodes[depot]["y"])
        graph = ox.graph_from_gdfs(nodes_to_explore, edges)
        nodes_to_explore, edges = ox.graph_to_gdfs(
            graph, edges=True, nodes=True, node_geometry=True
        )
        m = edges.explore(tiles="cartodbdarkmatter", cmap="plasma", column="weight")
        m = nodes_to_explore.explore(m=m, legend=True)

        Geocoder().add_to(m)
        m.save("test.html")
        print(f"Number of points found: {len(graph.nodes)}")

        return graph, depot
