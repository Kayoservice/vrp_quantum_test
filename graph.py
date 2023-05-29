from operator import itemgetter
from typing import Tuple

import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame, GeoSeries
from geopy import distance
from IPython.display import IFrame
from networkx import MultiDiGraph
from osmnx import geocoder
from shapely.geometry import Point, Polygon


class OSMGraph:
    def __init__(
        self,
        distance_kms: int,
        address: str,
    ) -> None:
        self.distance_kms = distance_kms
        self.address = address

    def add_interest_points(
        self, depot: str, graph: MultiDiGraph, delivery_points: GeoSeries
    ) -> "MultiDiGraph":
        node_id = 1
        for point in delivery_points:

            print(f"processing point: {point}")
            nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
            distance_found = distance.distance(
                (graph.nodes[nearest_node]["y"], graph.nodes[nearest_node]["x"]),
                (point.y, point.x),
            )
            print(f"distance from {nearest_node} for {node_id} is {distance_found}")
            # Add simulated delivery_point node
            graph.add_node(node_id, x=point.x, y=point.y, attr={"goal": "delivery"})

            # Add bidirectional edge
            graph.add_edge(nearest_node, node_id, weight=distance_found.m)
            graph.add_edge(node_id, nearest_node, weight=distance_found.m)

            node_id += 1
        return graph

    def generate_fake_delivery_graph(self) -> Tuple[GeoSeries, MultiDiGraph, str]:
        address = "La Madeleine,Nord,France"
        center_lat, center_long = geocoder.geocode(query=address)
        print(f"Starting point is lat:{center_lat} long:{center_long}")
        graph = ox.graph_from_place(
            address, network_type="drive", simplify=False, truncate_by_edge=True
        )
        (*_,) = ox.plot_graph(graph, bgcolor="b", show=False, close=False)
        depot = ox.distance.nearest_nodes(graph, center_lat, center_long)
        print(f"nearest node id from the center selected as a depot: {depot}")
        # graph=ox.project_graph(graph)
        delivery_points = ox.utils_geo.sample_points(graph.to_undirected(), 60)

        print("delivery points:", delivery_points)
        return delivery_points, graph.to_undirected(), depot

    def add_all_edge_from_entrepot(
        self, graph: MultiDiGraph, depot: str
    ) -> MultiDiGraph:
        for node in graph.nodes():
            print(f"processing point: {node}")
            distance_depot = distance.distance(
                (graph.nodes[depot]["y"], graph.nodes[depot]["x"]),
                (graph.nodes[node]["y"], graph.nodes[node]["x"]),
            )
            # add edge from entrepot
            graph.add_edge(depot, node, weight=distance_depot.m)
        return graph

    def generate_graph_from_address(self) -> Tuple[MultiDiGraph, str]:
        ox.config(log_console=True, use_cache=True)

        delivery_points, graph, depot = self.generate_fake_delivery_graph()

        print(
            f"bus stop found used as fake deliveries: {len(delivery_points.to_dict())}"
        )
        # Get the nearest nodes to bus stops
        bus_stop_nodes = list(map(itemgetter(1), delivery_points.index.values))
        print(bus_stop_nodes[:5])
        nodes = [depot] + bus_stop_nodes
        graph = ox.project_graph(graph)
        graph = ox.utils_graph.remove_isolated_nodes(graph)

        graph = ox.consolidate_intersections(
            graph, rebuild_graph=True, tolerance=15, dead_ends=False
        )
        print(nodes[:5])
        graph = self.add_interest_points(depot, graph, delivery_points)
        graph = self.add_all_edge_from_entrepot(graph, depot)

        # Get edges as GeoDataFrames
        nodes_to_explore, edges = ox.graph_to_gdfs(graph, edges=True, nodes=True)

        for _, edge in edges.fillna("").iterrows():
            weight = None
            if not edge["length"]:
                weight = 0
            else:
                weight = round(edge["length"])
            nx.set_edge_attributes(graph, weight, "weight")

        m = edges.explore(tiles="cartodbdarkmatter", cmap="plasma", column="length")
        m = nodes_to_explore.explore(m=m, color="pink")
        m.save("test.html")
        print(f"Number of points found: {len(nodes)}")

        return graph, depot
