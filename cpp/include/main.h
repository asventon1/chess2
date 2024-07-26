#include <pybind11/pybind11.h>

std::string minimax(const std::string &fen, const std::string &move, unsigned int depth, float extra_depth_i);

namespace py = pybind11;

PYBIND11_MODULE(cpp_part, mod) {
    mod.def("minimax", &minimax, "do minimax algoritm");
}
