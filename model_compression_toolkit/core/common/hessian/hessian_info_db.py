from model_compression_toolkit.core.common.hessian.trace_hessian_request import LightWeightRequest, TraceHessianRequest
import numpy as np

class HessianInfoDB:
    def __init__(self):
        self._lw_hessian_request_to_score_list = {}

    def _compute_ligthweight_request(self, request: TraceHessianRequest):
        return LightWeightRequest(mode=request.mode,
                                  granularity=request.granularity,
                                  target_node_name=request.target_node.name)

    def get_stored_info(self, request: TraceHessianRequest):
        lw_request = self._compute_ligthweight_request(request)
        return self._lw_hessian_request_to_score_list.get(lw_request, [])

    def add_to_stored_info(self, request: TraceHessianRequest, approximation: np.ndarray):
        lw_request = self._compute_ligthweight_request(request)
        if lw_request in self._lw_hessian_request_to_score_list:
            self._lw_hessian_request_to_score_list[lw_request].append(approximation)
        else:
            self._lw_hessian_request_to_score_list[lw_request] = [approximation]

    def clear_saved_hessian_info(self):
        """Clears the saved info approximations."""
        self._lw_hessian_request_to_score_list={}

    def count_saved_info_of_request(self, hessian_request:TraceHessianRequest) -> int:
        """
        Counts the saved approximations of Hessian info (traces, for now) for a specific request.
        If some approximations were computed for this request before, the amount of approximations (per image)
        will be returned. If not, zero is returned.

        Args:
            hessian_request: The request configuration for which to count the saved data.

        Returns:
            Number of saved approximations for the given request.
        """
        return len(self.get_stored_info(hessian_request))