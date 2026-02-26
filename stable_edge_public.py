import pandas as pd


# ==========================================================
# Data Loading
# ==========================================================

def load_edge_list(file_path):
    """
    Load an edge list file.

    Required columns:
        - source
        - target
        - cor
        - correlation
        - p_value
    """
    return pd.read_csv(file_path, sep="\t")


# ==========================================================
# Edge Processing
# ==========================================================

def create_edge_dict(df):
    """
    Convert edge list DataFrame into a dictionary
    using sorted node pairs as keys.
    """
    edge_dict = {}

    for _, row in df.iterrows():
        node1, node2 = sorted([row["source"], row["target"]])
        edge_key = (node1, node2)

        edge_dict[edge_key] = {
            "cor": row["cor"],
            "correlation": row["correlation"],
            "p_value": row["p_value"],
        }

    return edge_dict


def find_stable_edges(edges_dict_1, edges_dict_2):
    """
    Identify edges present in both networks
    with consistent correlation direction.
    """
    common_edges = set(edges_dict_1.keys()).intersection(
        set(edges_dict_2.keys())
    )

    stable_edges = []

    for edge_key in common_edges:
        cor1 = edges_dict_1[edge_key]["cor"]
        cor2 = edges_dict_2[edge_key]["cor"]

        if cor1 == cor2:
            stable_edges.append({
                "source": edge_key[0],
                "target": edge_key[1],
                "cor": cor1,
                "correlation_file1": edges_dict_1[edge_key]["correlation"],
                "correlation_file2": edges_dict_2[edge_key]["correlation"],
                "p_value_file1": edges_dict_1[edge_key]["p_value"],
                "p_value_file2": edges_dict_2[edge_key]["p_value"],
            })

    return pd.DataFrame(stable_edges)


# ==========================================================
# Main
# ==========================================================

def main(
    edge_file_1="c_edge_list.tsv",
    edge_file_2="o_edge_list.tsv",
    output_file="stable_edges.tsv"
):
    edges1 = load_edge_list(edge_file_1)
    edges2 = load_edge_list(edge_file_2)

    edges1_dict = create_edge_dict(edges1)
    edges2_dict = create_edge_dict(edges2)

    stable_df = find_stable_edges(
        edges1_dict,
        edges2_dict
    )

    stable_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    main()