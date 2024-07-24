cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp_part)

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")
set(CMAKE_BUILD_TYPE Release)

#add_library(cpp_part MODULE src/main.cpp)
pybind11_add_module(cpp_part MODULE src/main.cpp)
target_link_libraries(cpp_part PUBLIC "${TORCH_LIBRARIES}" pybind11::module)
#add_executable(cpp_part_exe src/main.cpp)
#target_link_libraries(cpp_part_exe PUBLIC "${TORCH_LIBRARIES}")


message(TORCH_LIBRARIES="${TORCH_LIBRARIES}")
message(TORCH_CXX_FLAGS="${TORCH_CXX_FLAGS}")
set_property(TARGET cpp_part PROPERTY CXX_STANDARD 17)
