cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(example-app-lib MODULE example-app.cpp)
target_link_libraries(example-app-lib PUBLIC "${TORCH_LIBRARIES}" pybind11::module)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app PUBLIC "${TORCH_LIBRARIES}" pybind11::embed)
message(TORCH_LIBRARIES="${TORCH_LIBRARIES}")
message(TORCH_CXX_FLAGS="${TORCH_CXX_FLAGS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
