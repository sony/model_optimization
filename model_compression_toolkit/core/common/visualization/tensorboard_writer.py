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

from copy import deepcopy

import io
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.event_pb2 import Event, TaggedRunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, NodeExecStats, DeviceStepStats, AllocatorMemoryUsed
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from typing import List, Any, Dict
from networkx import topological_sort
from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector

DEVICE_STEP_STATS = "/device:CPU:0"


def get_node_properties(node_dict_to_log: dict,
                        output_shapes: List[tuple] = None) -> Dict[str, Any]:
    """
    Create a dictionary with properties for a node to display.

    Args:
        node_dict_to_log: Node's attributes to display.
        output_shapes: Output shapes of the node.

    Returns:
        Dictionary with the node's properties.

    """

    node_properties = {}
    if node_dict_to_log is not None:
        # Add a key-value pair by its type
        for k, v in node_dict_to_log.items():
            if type(v) == int:
                node_properties[str(k)] = AttrValue(i=v)
            elif type(v) == float:
                node_properties[str(k)] = AttrValue(f=v)
            elif type(v) == bool:
                node_properties[str(k)] = AttrValue(b=v)
            else:
                node_properties[str(k)] = AttrValue(s=str(v).encode("utf-8"))

    # Create protobuf for the node's output shapes
    if output_shapes is not None:
        tshape_protos = []
        for output_shape in output_shapes:  # create protobuf for each output shape
            proto_dims_list = []
            for dim in output_shape:
                proto_dims_list.append(TensorShapeProto.Dim(size=dim))
            tshape_proto = TensorShapeProto(dim=proto_dims_list)
            tshape_protos.append(tshape_proto)
        node_properties['_output_shapes'] = AttrValue(list=AttrValue.ListValue(shape=tshape_protos))
    return node_properties


class TensorboardWriter(object):
    """
    Class to log events to display using Tensorboard such as graphs, histograms, images, etc.
    """

    def __init__(self, dir_path: str, fw_info: FrameworkInfo):
        """
        Initialize a TensorboardWriter object.
        
        Args:
            dir_path: Path to save all events to display on Tensorboard.
            fw_info: FrameworkInfo object (needed for computing nodes' weights memory).

        """
        self.dir_path = dir_path
        # we hold EventWriter per tag name, so events can be gathered by tags (like phases during the quantization
        # process).
        self.tag_name_to_event_writer = {}
        self.fw_info = fw_info

    def close(self):
        """

        Close all event-writers the TensorboardWriter holds.
        Should be called at the end of logging process.

        """
        for writer in self.tag_name_to_event_writer.values():
            writer.close()

    def add_histograms(self, graph: Graph, main_tag_name: str):
        """
        Add histograms to display on Tensorboard. All existing histograms in a graph are 
        logged with a tag name main_tag_name.
        
        Args:
            graph: Graph to display all collected histograms it contains.
            main_tag_name: Tag to attach to all histograms.

        """

        def __create_hist_proto(bins: np.ndarray,
                                counts: np.ndarray) -> HistogramProto:
            """
            Create a protobuf for a histogram using collected bins and counts.

            Args:
                bins: Values of histogram bins.
                counts: Counts of histogram values.

            Returns:
                Protobuf of a histogram to display on Tensorboard.
            """

            sum_sq = ((bins * bins) * counts).sum()
            return HistogramProto(min=bins.min(),
                                  max=bins.max(),
                                  num=len(bins),
                                  sum=(bins * counts).sum(),
                                  sum_squares=sum_sq,
                                  bucket_limit=bins.tolist(),
                                  bucket=counts.tolist())

        def __create_histo_event(statistics_collector: BaseStatsCollector):
            """
            Create an event of histogram, and attach it to a list of events outside
            the scope called 'events'.

            Args:
                statistics_collector: Statistics collector to create an event from its histogram.

            """
            if statistics_collector.require_collection():
                if hasattr(statistics_collector, 'hc'):
                    if statistics_collector.hc.is_legal:
                        bins, counts = statistics_collector.hc.get_histogram()
                        if bins is not None and counts is not None:
                            hist = __create_hist_proto(bins[:-1], counts)
                            summary = Summary(value=[Summary.Value(tag=n.name, histo=hist)])
                            events.append(Event(summary=summary))

        events = []
        for n in graph.nodes:
            collector = graph.get_out_stats_collector(n)
            if collector is not None:
                statistics = graph.get_out_stats_collector(n)
                if isinstance(statistics, list):
                    for s in statistics:
                        __create_histo_event(s)
                else:
                    __create_histo_event(statistics)

        # Get the event writer for this tag name
        er = self.__get_event_writer_by_tag_name(main_tag_name)

        for event in events:
            er.add_event(event)
        er.flush()

    def add_graph(self,
                  graph: Graph,
                  main_tag_name: str):
        """
        Add a graph to display on Tensorboard. The graph is tagged with the name main_tag_name.

        Args:
            graph: Graph to display on Tensorboard.
            main_tag_name: Tag to attach to the graph.

        """

        def __get_node_act_attr(n: BaseNode) -> Dict[str, Any]:
            """
            Create a dictionary to display as the node's attributes.
            The dictionary contains information from node's activation attributes.

            Args:
                n: Node to create its attributes.

            Returns:
                Dictionary containing attributes to display.
            """
            attr = dict()
            if n.final_activation_quantization_cfg is not None:
                attr.update(n.final_activation_quantization_cfg.__dict__)
            elif n.candidates_quantization_cfg is not None:
                attr.update(n.get_unified_activation_candidates_dict())
            return attr

        def __get_node_weights_attr(n: BaseNode) -> Dict[str, Any]:
            """
            Create a dictionary to display as the node's attributes.
            The dictionary contains information from node's weights attributes.

            Args:
                n: Node to create its attributes.

            Returns:
                Dictionary containing attributes to display.
            """
            # To log quantization configurations we need to check
            # if they exist at all, as we can log the initial graph,
            # which its nodes do not have configurations yet.
            # Log final config or unified candidates, not both
            attr = dict()
            if n.final_weights_quantization_cfg is not None:
                attr.update(n.final_weights_quantization_cfg.__dict__)
            elif n.candidates_quantization_cfg is not None:
                attr.update(n.get_unified_weights_candidates_dict())
            return attr

        def __get_node_attr(n: BaseNode) -> Dict[str, Any]:
            """
            Create a dictionary to display as the node's attributes.
            The dictionary contains information from node's framework attributes and quantization attributes

            Args:
                n: Node to create its attributes.

            Returns:
                Dictionary containing attributes to display.
            """
            attr = deepcopy(n.framework_attr)
            if n.quantization_attr is not None:
                attr.update(n.quantization_attr)
            return attr

        def __get_node_output_dims(n: BaseNode) -> List[tuple]:
            """
            Get node's output shapes. If the first dimension in an output shape is None,
            it means the batch size is dynamic, and it's replaced with -1 to mark it.

            Args:
                n: Node to get its output shapes.

            Returns:
                A list of tuples where each tuple is an output shape of the node.
            """

            # For nodes with an "empty" output shape.
            output_shape = (None,) if n.output_shape == () else n.output_shape

            dims = []
            if isinstance(output_shape, list):
                for o in output_shape:
                    shape_wo_none = (-1,) + o[1:] if o[0] is None else o
                    dims.append(shape_wo_none)
            else:
                dims = [(-1,) + output_shape[1:] if output_shape[0] is None else output_shape]
            return dims

        def __create_node_stats(n: BaseNode):
            """
            Create a NodeExecStats for a node in the graph. A NodeExecStats contains the
            memory and compute time a node requires.

            Args:
                n: Node to create its NodeExecStats.

            Returns:
                A NodeExecStats for a node containing its memory and compute time.

            """

            return NodeExecStats(node_name=n.name,
                                 memory=[AllocatorMemoryUsed(
                                     total_bytes=int(n.get_memory_bytes(self.fw_info))
                                 )])

        graph_def = GraphDef()  # GraphDef to add to Tensorboard

        node_stats = []
        types_dict = dict()
        node_sort = list(topological_sort(graph))
        for n in node_sort:  # For each node in the graph, we create NodeDefs and connect them to existing NodeDefs
            # ----------------------------
            # Main NodeDef: framework attributes
            # ----------------------------
            main_node_def = NodeDef(attr=get_node_properties(__get_node_attr(n), __get_node_output_dims(n)))
            main_node_def.device = n.type.__name__  # For coloring different ops differently
            main_node_def.op = n.type.__name__
            op_id = types_dict.get(main_node_def.op, 0)
            if len(graph.incoming_edges(n)) == 0:  # Input layer
                n.tb_node_def = 'Input/' + n.name
            elif len(graph.out_edges(n)) == 0:  # Output layer
                n.tb_node_def = 'Output/' + n.name
            else:
                n.tb_node_def = graph.name + '/' + main_node_def.op + '_' + str(op_id) + '/' + n.name
            main_node_def.name = n.tb_node_def
            for e in graph.incoming_edges(n):  # Connect node to its incoming nodes
                i_tensor = f'{e.source_node.tb_node_def}:{e.source_index}'
                main_node_def.input.append(i_tensor)
            # ----------------------------
            # Weights NodeDef
            # ----------------------------
            attr = __get_node_weights_attr(n)
            if bool(attr):
                weights_node_def = NodeDef(attr=get_node_properties(attr))
                weights_node_def.name = main_node_def.name + ".weights"
                main_node_def.input.append(f'{weights_node_def.name}:{1}')
                graph_def.node.extend([weights_node_def])  # Add the node to the graph
            # ----------------------------
            # Activation NodeDef
            # ----------------------------
            attr = __get_node_act_attr(n)
            if bool(attr):
                act_node_def = NodeDef(attr=get_node_properties(attr, __get_node_output_dims(n)))
                n.tb_node_def = main_node_def.name + ".activation"
                act_node_def.name = n.tb_node_def
                act_node_def.input.append(f'{main_node_def.name}:{0}')
                graph_def.node.extend([act_node_def])  # Add the node to the graph

            graph_def.node.extend([main_node_def])  # Add the node to the graph
            node_stats.append(__create_node_stats(n))
            types_dict.update({main_node_def.op: op_id + 1})

        er = self.__get_event_writer_by_tag_name(main_tag_name)
        event = Event(graph_def=graph_def.SerializeToString())
        er.add_event(event)

        # Logging nodes memory and computation time statistics
        stepstats = RunMetadata(step_stats=StepStats(
            dev_stats=[DeviceStepStats(device=DEVICE_STEP_STATS, node_stats=node_stats)])
        )

        trm = TaggedRunMetadata(tag='Resources', run_metadata=stepstats.SerializeToString())
        event = Event(tagged_run_metadata=trm)
        er.add_event(event)
        er.flush()

    def __get_event_writer_by_tag_name(self,
                                       main_tag_name: str) -> EventFileWriter:
        """
        Retrieve an EventWriter by a tag name from the mapping the TensorboardWriter holds.

        Args:
            main_tag_name: Tag name to retrieve its EventWriter.

        Returns:
            EventFileWriter corresponding to the tag name.
        """

        if main_tag_name in self.tag_name_to_event_writer:  # if an EventWriter already exists, get it
            er = self.tag_name_to_event_writer.get(main_tag_name)
        else:  # if not - create such an EventWriter and save it
            er = EventFileWriter(f'{self.dir_path}/{main_tag_name}')
            self.tag_name_to_event_writer[main_tag_name] = er
        return er

    def add_min_max(self, graph: Graph, main_tag_name: str):
        """
        Add min/max per channel to display on Tensorboard.
        All existing MinMaxPerChannelCollector in a graph are logged with a tag name main_tag_name.

        Args:
            graph: Graph to display all collected MinMaxPerChannelCollectors it contains.
            main_tag_name: Tag to attach to all MinMaxPerChannelCollectors.

        """
        min_events = []
        max_events = []
        for n in graph.nodes:
            collector = graph.get_out_stats_collector(n)
            if collector is not None:
                if hasattr(collector, 'mpcc'):
                    if collector.mpcc.is_legal:
                        mpcc = deepcopy(collector.mpcc)
                        min_pc = mpcc.min_per_channel
                        max_pc = mpcc.max_per_channel
                        for i in range(len(min_pc)):
                            min_i = min_pc[i]
                            max_i = max_pc[i]
                            # use step for channel index as we log the min/max per channel
                            min_events.append(Event(step=i, summary=Summary(
                                value=[Summary.Value(tag=n.name, simple_value=min_i)])))
                            max_events.append(Event(step=i, summary=Summary(
                                value=[Summary.Value(tag=n.name, simple_value=max_i)])))

        # Use a new tag to include both main tag and a 'min_per_channel' tag.
        er = self.__get_event_writer_by_tag_name(main_tag_name + '/min_per_channel')

        for e in min_events:
            er.add_event(e)

        # Use a new tag to include both main tag and a 'max_per_channel' tag.
        er = self.__get_event_writer_by_tag_name(main_tag_name + '/max_per_channel')
        for e in max_events:
            er.add_event(e)

        er.flush()

    def add_mean(self, graph: Graph, main_tag_name: str):
        """
        Add mean per channel to display on Tensorboard.
        All existing MeanCollectors in a graph are logged with a tag name main_tag_name.

        Args:
            graph: Graph to display all collected MeanCollectors it contains.
            main_tag_name: Tag to attach to all MeanCollectors.

        """
        mean_events = []
        for n in graph.nodes:
            collector = graph.get_out_stats_collector(n)
            if collector is not None:
                if hasattr(collector, 'mc'):
                    if collector.mc.is_legal:
                        mc = deepcopy(collector.mc)
                        mean_pc = mc.state
                        for i in range(len(mean_pc)):
                            mean_i = mean_pc[i]
                            # use step for channel index as we log the mean per channel
                            mean_events.append(Event(step=i, summary=Summary(
                                value=[Summary.Value(tag=n.name, simple_value=mean_i)])))

        # Get the event writer for this tag name
        er = self.__get_event_writer_by_tag_name(main_tag_name + '/mean_per_channel')

        for e in mean_events:
            er.add_event(e)

        er.flush()

    def add_all_statistics(self, graph: Graph, main_tag_name: str):
        """
        Add all collected statistics to display on Tensorboard.
        All existing collectors in a graph are logged with a tag name main_tag_name.

        Args:
            graph: Graph to display all collectors it contains.
            main_tag_name: Tag to attach to all collectors display.

        """
        self.add_histograms(graph, main_tag_name)
        self.add_min_max(graph, main_tag_name)
        self.add_mean(graph, main_tag_name)

    def add_figure(self,
                   figure: Figure,
                   figure_tag: str,
                   main_tag_name: str = 'figures'):
        """
        Add matplotlib figure to display. The figure tag is combined from a main_main_tag_name
        and a specific figure_tag for that figure.

        Args:
            figure: Matplotlib figure to display.
            figure_tag: Tag to add to the figure.
            main_tag_name: Main tag which the figure is tagged under.

        """
        figure.canvas.draw()
        data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

        h, w, c = data.shape
        output = io.BytesIO()
        Image.fromarray(data).save(output, format='PNG')

        img_summary = Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=output.getvalue())
        output.close()

        event = Event(summary=Summary(value=[Summary.Value(tag=figure_tag, image=img_summary)]))

        # Get the event writer for this tag name
        er = self.__get_event_writer_by_tag_name(main_tag_name)
        er.add_event(event)
        er.flush()
