import tracksdata as td


def load_gt(sample_name, train_dir="../input/train"):
    """
    Load the ground truth data for a given sample.

    Args:
        sample_name (str): The name of the sample.
        train_dir (str, Optional): The directory containing the training data. Defaults to "../input/train".

    Returns:
        nodes (polars.DataFrame): The nodes of the graph.
        edges (polars.DataFrame): The edges of the graph.
        estimated_number_of_nodes (int): The estimated number of nodes in the graph.
    """
    result = td.graph.IndexedRXGraph.from_geff(f"{str(train_dir)}/{sample_name}.geff")
    g, meta = result
    estimated_number_of_nodes = meta.extra['estimated_number_of_nodes']
    edges = g.edge_attrs()
    nodes = g.node_attrs()
    return nodes, edges, estimated_number_of_nodes
