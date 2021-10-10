# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import json
import os
from typing import Any, Dict, Tuple

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.node import Node


def check_str(in_sting: str) -> bool:
    """
    Checks if an open bracket is in a string.

    Args:
        in_sting: String to check.

    Returns:
        Whether an open bracket is in the string or not.
    """

    return '(' not in in_sting and '[' not in in_sting and '{' not in in_sting


def node_dict(n: Node) -> Dict[str, Any]:
    """
    Get a dictionary with a node's attributes for displaying.

    Args:
        n: Node to get its attributes to display.

    Returns:
        A dictionary with params of the node to display when visualizing the graph.
    """

    framework_attr = {k: str(v) for k, v in n.framework_attr.items() if check_str(str(v))}
    framework_attr.update({k: str(v) for k, v in n.quantization_attr.items() if check_str(str(v))})
    framework_attr.update({'op': n.layer_class.__name__})

    if n.quantization_cfg is not None:
        for k, v in n.quantization_cfg.activation_quantization_params.items():
            framework_attr.update({k: str(v)})
        framework_attr.update({'activation_is_signed': str(n.quantization_cfg.activation_is_signed)})

    return {"id": n.name,
            "group": n.layer_class.__name__,
            "label": n.layer_class.__name__,
            "title": "",
            "properties": framework_attr}


def edge_dict(i: int,
              edge: Tuple[Node, Node],
              graph):
    """
    Create a dictionary of attributes to visualize an edge in the graph.

    Args:
        i: Edge's ID.
        edge: Tuple of two nodes (source and destination).
        graph: Graph the edge is in.

    Returns:
        Dictionary of attributes to visualize an edge in a graph.
    """

    return {"id": str(i),
            "from": edge[0].name,
            "to": edge[1].name,
            "label": "",
            "title": "",
            "properties": graph.get_edge_data(edge[0], edge[1])}


def write_vis_graph(folder_path: str,
                    file_name: str,
                    graph):
    """
    Create and save a json file containing data about the graph to visualize (such as nodes and edges).
    folder_path and file_name determine where the json file will be saved.

    Args:
        folder_path: Dir path to save the json file.
        file_name: File name of the json file.
        graph: Graph to visualize.

    """

    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name + '.json')

    nodes_list = [node_dict(n) for n in graph.nodes()]
    edges_list = [edge_dict(i, e, graph) for i, e in enumerate(graph.edges())]
    graph_dict = {'nodes': nodes_list,
                  'edges': edges_list,
                  "options": {}
                  }

    with open(file_path, 'w') as json_file:
        json.dump(graph_dict, json_file)

    common.Logger.info(f"Writing Vis Graph to:{file_path}")
