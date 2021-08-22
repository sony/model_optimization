# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import json
import os
from typing import Any, Dict, Tuple

from network_optimization_package import common
from network_optimization_package.common.graph.node import Node


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
