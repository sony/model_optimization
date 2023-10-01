from typing import List, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tqdm import tqdm
import numpy as np

from model_compression_toolkit.constants import MIN_JACOBIANS_ITER, JACOBIANS_COMP_TOLERANCE, EPS
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig, HessianMode, HessianGranularity
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.core.keras.hessian.hessian_calculator_keras import HessianCalculatorKeras
from model_compression_toolkit.logger import Logger
from tensorflow.python.util.object_identity import Reference as TFReference


class ActivationHessianCalculatorKeras(HessianCalculatorKeras):

    def __init__(self,
                 graph: Graph,
                 config: HessianConfig,
                 input_images: List[tf.Tensor],
                 fw_impl):

        super(ActivationHessianCalculatorKeras, self).__init__(graph=graph,
                                                               config=config,
                                                               input_images=input_images,
                                                               fw_impl=fw_impl)

    def compute(self):
        if self.config.granularity==HessianGranularity.PER_LAYER:
            output_list = self._get_model_output_replacement()
            all_outputs_indices=[]
            if self.config.search_output_replacement:
                all_outputs_indices = self._update_ips_with_outputs_replacements(output_list,
                                                                                 self.config.nodes_names_for_hessian_computation)

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                outputs, interest_points_tensors = self._get_model_outputs_for_single_image(output_list,
                                                                                            gradient_tape=g)
                # Concat outputs
                # First, we need to unfold all outputs that are given as list, to extract the actual output tensors
                unfold_outputs = []
                for output in outputs:
                    if isinstance(output, List):
                        unfold_outputs += output
                    else:
                        unfold_outputs.append(output)

                r_outputs = [tf.reshape(output, shape=[output.shape[0], -1]) for output in unfold_outputs]

                concat_axis_dim = [o.shape[0] for o in r_outputs]
                if not all(d == concat_axis_dim[0] for d in concat_axis_dim):
                    Logger.critical(
                        "Can't concat model's outputs for gradients calculation since the shape of the first axis "  # pragma: no cover
                        "is not equal in all outputs.")

                output = tf.concat(r_outputs, axis=1)

                ipts_jac_trace_approx = []
                for ipt in tqdm(interest_points_tensors):  # Per Interest point activation tensor
                    trace_jv = []
                    for j in range(self.config.num_iterations):  # Approximation iterations
                        # Getting a random vector with normal distribution
                        v = tf.random.normal(shape=output.shape)
                        f_v = tf.reduce_sum(v * output)

                        with g.stop_recording():
                            # Computing the jacobian approximation by getting the gradient of (output * v)
                            jac_v = g.gradient(f_v, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                            jac_v = tf.reshape(jac_v, [jac_v.shape[0], -1])
                            jac_trace_approx = tf.reduce_mean(tf.reduce_sum(tf.pow(jac_v, 2.0)))

                            # If the change to the mean Jacobian approximation is insignificant we stop the calculation
                            if j > MIN_JACOBIANS_ITER:
                                new_mean = np.mean([jac_trace_approx, *trace_jv])
                                delta = new_mean - np.mean(trace_jv)
                                if np.abs(delta) / (np.abs(new_mean) + 1e-6) < JACOBIANS_COMP_TOLERANCE:
                                    trace_jv.append(jac_trace_approx)
                                    break

                            trace_jv.append(jac_trace_approx)
                    ipts_jac_trace_approx.append(2 * tf.reduce_mean(trace_jv) / output.shape[
                        -1])  # Get averaged squared jacobian trace approximation

                ipts_jac_trace_approx = tf.reduce_mean([ipts_jac_trace_approx], axis=0)  # Just to get one tensor instead of list of tensors with single element

                if self.config.norm_weights:
                    normalized_ipts_jac_trace_approx = self._normalize_weights(ipts_jac_trace_approx,
                                              all_outputs_indices,
                                              self.config.alpha)
                    return self._attach_interst_point_names_to_scores(normalized_ipts_jac_trace_approx)
                else:
                    return self._attach_interst_point_names_to_scores(ipts_jac_trace_approx)
        else:
            raise NotImplemented


    def _attach_interst_point_names_to_scores(self, scores: List[float]):
        res = {}
        assert len(self.config.nodes_names_for_hessian_computation)==len(scores)
        for point_name, score in zip(self.config.nodes_names_for_hessian_computation, scores):
            res[point_name]=score
        return res

    def _update_ips_with_outputs_replacements(self,
                                              outputs_replacement_nodes,
                                              interest_points):
        """
        Updates the list of interest points with the set of pre-calculated replacement outputs.
        Also, returns the indices of all output nodes (original, replacements and nodes in between them) in a
        topological sorted interest points list (for later use in gradients computation and normalization).

        Returns: A list of indices of the output nodes in the sorted interest points list.

        """
        # todo: make sure in GPTQ outputs_replacement_nodes is an empty list (more specificaly the returned list from this function should be an empty list

        replacement_outputs_to_ip = [r_node for r_node in outputs_replacement_nodes if
                                     r_node not in interest_points]
        updated_interest_points = interest_points + replacement_outputs_to_ip

        # Re-sort interest points in a topological order according to the graph's sort
        interest_points = [n for n in self.graph.get_topo_sorted_nodes() if n in updated_interest_points]

        output_indices = [interest_points.index(n.node) for n in self.graph.get_outputs()]
        replacement_indices = [interest_points.index(n) for n in outputs_replacement_nodes]
        return list(set(output_indices + replacement_indices))


    def _normalize_weights(self,
                           jacobians_traces: List,
                           all_outputs_indices: List[int],
                           alpha: float) -> List[float]:
        """
        Output layers or layers that come after the model's considered output layers,
        are assigned with a constant normalized value, according to the given alpha variable and the number of such
        layers.
        Other layers returned weights are normalized by dividing the jacobian-based weights value by the sum of all
        other values.

        Args:
            jacobians_traces: The approximated average jacobian-based weights of each interest point.
            all_outputs_indices: A list of indices of all nodes that consider outputs.
            alpha: A multiplication factor.

        Returns: Normalized list of jacobian-based weights (for each interest point).

        """

        sum_without_outputs = sum(
            [jacobians_traces[i] for i in range(len(jacobians_traces)) if i not in all_outputs_indices])
        normalized_grads_weights = [self._get_normalized_weight(grad,
                                                                i,
                                                                sum_without_outputs,
                                                                all_outputs_indices,
                                                                alpha)
                                    for i, grad in enumerate(jacobians_traces)]

        return normalized_grads_weights

    def _get_normalized_weight(self,
                               grad: float,
                               i: int,
                               sum_without_outputs: float,
                               all_outputs_indices: List[int],
                               alpha: float) -> float:
        """
        Normalizes the node's gradient value. If it is an output or output replacement node than the normalized value is
        a constant, otherwise, it is normalized by dividing with the sum of all gradient values.

        Args:
            grad: The gradient value.
            i: The index of the node in the sorted interest points list.
            sum_without_outputs: The sum of all gradients of nodes that are not considered outputs.
            all_outputs_indices: A list of indices of all nodes that consider outputs.
            alpha: A multiplication factor.

        Returns: A normalized jacobian-based weights.

        """

        if i in all_outputs_indices:
            return alpha / len(all_outputs_indices)
        else:
            return ((1 - alpha) * grad / (sum_without_outputs + EPS)).numpy()

    def _get_model_output_replacement(self) -> List[str]:
        """
        If a model's output node is not compatible for the task of gradients computation we need to find a predecessor
        node in the model's graph representation which is compatible and use it for the gradients' computation.
        This method searches for this predecessor node for each output of the model.

        Returns: A list of output replacement nodes.

        """

        replacement_outputs = []
        for n in self.graph.get_outputs():
            prev_node = n.node
            while not self.fw_impl.is_node_compatible_for_metric_outputs(prev_node):
                prev_node = self.graph.get_prev_nodes(prev_node)
                assert len(prev_node) == 1, "A none compatible output node has multiple inputs, " \
                                            "which is incompatible for metric computation."
                prev_node = prev_node[0]
            replacement_outputs.append(prev_node)
        return replacement_outputs

    def _get_model_outputs_for_single_image(self,
                                   output_list: List[str],
                                   gradient_tape: tf.GradientTape) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """
        Computes the model's output according to the given graph representation on the given input,
        while recording necessary intermediate tensors for gradients computation.

        Args:
            graph_float: Graph to build its corresponding Keras model.
            model_input_tensors: A mapping between model input nodes to an input batch.
            interest_points: List of nodes which we want to get their feature map as output, to calculate distance
            metric.
            output_list: List of nodes that considered as model's output for the purpose of gradients computation.
            gradient_tape: A GradientTape object for recording necessary info for computing gradients.

        Returns: A list of output tensors and a list of activation tensors of all interest points.

        """
        model_input_tensors = {inode: self.fw_impl.to_tensor(self.input_images[i]) for i, inode in enumerate(self.graph.get_inputs())}

        node_to_output_tensors_dict = dict()

        # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
        oh = OperationHandler(self.graph)
        input_nodes_to_input_tensors = {inode: tf.convert_to_tensor(model_input_tensors[inode]) for
                                        inode in self.graph.get_inputs()}  # Cast numpy array to tf.Tensor

        interest_points_tensors = []
        output_tensors = []
        for n in oh.node_sort:
            # Build a dictionary from node to its output tensors, by applying the layers sequentially.
            op_func = oh.get_node_op_function(n)  # Get node operation function

            input_tensors = self._build_input_tensors_list(n,
                                                     self.graph,
                                                     node_to_output_tensors_dict)  # Fetch Node inputs
            out_tensors_of_n = self._run_operation(n,  # Run node operation and fetch outputs
                                             input_tensors,
                                             op_func,
                                             input_nodes_to_input_tensors)

            # Gradients can be computed only on float32 tensors
            if isinstance(out_tensors_of_n, list):
                for i, t in enumerate(out_tensors_of_n):
                    out_tensors_of_n[i] = tf.dtypes.cast(t, tf.float32)
            else:
                out_tensors_of_n = tf.dtypes.cast(out_tensors_of_n, tf.float32)

            print(n.name)
            print([ip.name for ip in self.config.nodes_names_for_hessian_computation])

            if n.name in [ip.name for ip in self.config.nodes_names_for_hessian_computation]:
                # Recording the relevant feature maps onto the gradient tape
                gradient_tape.watch(out_tensors_of_n)
                interest_points_tensors.append(out_tensors_of_n)
            if n in output_list:
                output_tensors.append(out_tensors_of_n)

            if isinstance(out_tensors_of_n, list):
                node_to_output_tensors_dict.update({n: out_tensors_of_n})
            else:
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

        return output_tensors, interest_points_tensors

    def _build_input_tensors_list(self,
                                  node: BaseNode,
                                 graph: Graph,
                                 node_to_output_tensors_dict: Dict[BaseNode, List[TFReference]]) -> List[
        List[TFReference]]:
        """
        Given a node, build a list of input tensors the node gets. The list is built
        based on the node's incoming edges and previous nodes' output tensors.

        Args:
            node: Node to build its input tensors list.
            graph: Graph the node is in.
            node_to_output_tensors_dict: A dictionary from a node to its output tensors.

        Returns:
            A list of the node's input tensors.
        """

        input_tensors = []
        # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
        # Append them in a result list.
        for ie in graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
            _input_tensors = [node_to_output_tensors_dict[ie.source_node][ie.source_index]]
            input_tensors.append(_input_tensors)
        return input_tensors

    def _run_operation(self,
                      n: BaseNode,
                      input_tensors: List[List[TFReference]],
                      op_func: Layer,
                      input_nodes_to_input_tensors: Dict[BaseNode, Any]) -> List[TFReference]:
        """
        Applying the layer (op_func) to the input tensors (input_tensors).

        Args:
            n: The corresponding node of the layer it runs.
            input_tensors: List of references to Keras tensors that are the layer's inputs.
            op_func: Layer to apply to the input tensors.
            input_nodes_to_input_tensors: A dictionary from a node to its input tensors.

        Returns:
            A list of references to Keras tensors. The layer's output tensors after applying the
            layer to the input tensors.
        """

        if len(input_tensors) == 0:  # Placeholder handling
            out_tensors_of_n = input_nodes_to_input_tensors[n]
        else:
            input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
            # Build a functional node using its args
            if isinstance(n, FunctionalNode):
                if n.inputs_as_list:  # If the first argument should be a list of tensors:
                    out_tensors_of_n = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
                else:  # If the input tensors should not be a list but iterated:
                    out_tensors_of_n = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:
                # If operator expects a single input tensor, it cannot be a list as it should have a dtype field.
                if len(input_tensors) == 1:
                    input_tensors = input_tensors[0]
                out_tensors_of_n = op_func(input_tensors)

        return out_tensors_of_n




