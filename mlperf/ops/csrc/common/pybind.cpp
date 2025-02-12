#include <vector>
#include <torch/extension.h>
#include "moegroupedgemm.hpp"
namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moegroupedgemm_gelu_forward", &moegroupedgemm_gelu_forward, "moegroupedgemm_forward");
    m.def("moegroupedgemm_gelu_backward", &moegroupedgemm_gelu_backward, "moegroupedgemm_backward");
    m.def("moegroupedgemm_forward", &moegroupedgemm_forward, "moegroupedgemm_forward");
    m.def("moegroupedgemm_backward", &moegroupedgemm_backward, "moegroupedgemm_backward");
}