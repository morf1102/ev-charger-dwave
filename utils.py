import matplotlib.pyplot as plt
import networkx as nx

def output_image(G, pois, charging_stations, new_charging_nodes=[]):
    """Create output image of solution scenario.

    Args:
        G (networkx graph): Grid graph of size w by h
        pois (list of tuples of ints): A fixed set of points of interest
        charging_stations (list of tuples of ints):
            A fixed set of current charging locations
        new_charging_nodes (list of tuples of ints):
            Locations of new charging stations

    Returns:
        None. Output saved to file "map.png".
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    #fig.suptitle("New EV Charger Locations")
    pos = {x: [x[0], x[1]] for x in G.nodes()}

    # Locate POIs in map
    poi_graph = G.subgraph(pois)

    # Locate old charging stations in map
    cs_graph = G.subgraph(charging_stations)

    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="k",
        node_size=3,
    )
    nx.draw_networkx(
        poi_graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="b",
        node_size=75,
    )
    nx.draw_networkx(
        cs_graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="r",
        node_size=75,
    )
    if len(new_charging_nodes)>0:
        if isinstance(new_charging_nodes[0], list):
            new_charging_nodes = [tuple(x) for x in new_charging_nodes]
        new_cs_graph = G.subgraph(new_charging_nodes)
        nx.draw_networkx(
            new_cs_graph,
            ax=ax,
            pos=pos,
            with_labels=False,
            node_color="g",
            node_size=75,
        )

    # Save image
    plt.show()


def distance(a, b):
    """
    Calculate the distance between two points.
    """
    return (a[0] ** 2 - 2 * a[0] * b[0] + b[0] ** 2) + (
        a[1] ** 2 - 2 * a[1] * b[1] + b[1] ** 2
    )