cmake_minimum_required(VERSION 3.14)
project(sh_baker_main CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-flto=auto>
  $<$<CONFIG:Release>:-march=native>
)
add_link_options(
  $<$<CONFIG:Release>:-flto=auto>
  $<$<CONFIG:Release>:-O3>
)

include(FetchContent)

# Enable offline build support:
# 1. Store dependencies in a persistent 'deps' folder within the source tree.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/deps")
# 2. Prevent FetchContent from attempting to connect to the internet to check for updates.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# --- Dependencies ---
# xatlas
FetchContent_Declare(
  xatlas
  GIT_REPOSITORY https://github.com/jpcy/xatlas.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(xatlas)

# mikktspace
FetchContent_Declare(
  mikktspace
  GIT_REPOSITORY https://github.com/mmikk/MikkTSpace.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(mikktspace)

# glog
find_package(glog REQUIRED)

# gflags
find_package(gflags REQUIRED)

# Googletest
find_package(GTest REQUIRED)

# stb_image
find_path(STB_IMAGE_INCLUDE_DIR NAMES stb_image.h PATH_SUFFIXES stb)
if(NOT STB_IMAGE_INCLUDE_DIR)
  message(FATAL_ERROR "stb_image not found")
endif()
add_library(gstb_image INTERFACE)
target_include_directories(gstb_image INTERFACE ${STB_IMAGE_INCLUDE_DIR})

# nlohmann_json
find_path(JSON_INCLUDE_DIR NAMES nlohmann/json.hpp)
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "nlohmann/json.hpp not found")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${JSON_INCLUDE_DIR})

# tinygltf
find_path(TINYGLTF_INCLUDE_DIR NAMES tiny_gltf.h)
if(NOT TINYGLTF_INCLUDE_DIR)
  message(FATAL_ERROR "tinygltf not found")
endif()
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${TINYGLTF_INCLUDE_DIR})
target_link_libraries(tinygltf INTERFACE gstb_image nlohmann_json)

# GLFW
find_package(glfw3 REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# Embree (Expected to be installed on system)
find_package(embree 4 REQUIRED)

# Eigen3 (Expected to be installed on system)
find_package(Eigen3 REQUIRED)

# TBB
find_package(TBB REQUIRED)

# --- Library (sh_baker) ---

set(LIB_SOURCES
    src/loader.cpp
    src/scene.cpp
    src/occlusion.cpp
    src/sh_coeffs.cpp
    src/saver.cpp
    src/baker.cpp
    src/atlas.cpp
    ${xatlas_SOURCE_DIR}/source/xatlas/xatlas.cpp
    ${mikktspace_SOURCE_DIR}/mikktspace.c
    src/rasterizer.cpp
    src/material.cpp
    src/dilation.cpp
    src/tinyexr_impl.cpp
    src/miniz.c
    src/light.cpp
)

set(LIB_HEADERS
    src/colorspace.h
    src/scene.h
    src/loader.h
    src/occlusion.h
    src/sh_coeffs.h
    src/saver.h
    src/baker.h
    src/rasterizer.h
    src/dilation.h
    src/material.h
    src/atlas.h
)

add_library(sh_baker SHARED ${LIB_SOURCES} ${LIB_HEADERS})

target_include_directories(sh_baker PUBLIC 
    src 
    ${xatlas_SOURCE_DIR}/source/xatlas
    ${mikktspace_SOURCE_DIR}
)

target_link_libraries(sh_baker PUBLIC
    tinygltf
    gstb_image
    glog::glog
    gflags
    embree
    Eigen3::Eigen
    TBB::tbb
)

# --- Executable (sh_baker_main) ---

add_executable(sh_baker_main src/main.cpp)
target_link_libraries(sh_baker_main PRIVATE sh_baker)

# --- Tests ---

enable_testing()

add_executable(sh_baker_test
    src/loader_test.cpp
    src/scene_test.cpp
    src/occlusion_test.cpp
    src/sh_coeffs_test.cpp
    src/saver_test.cpp
    src/baker_test.cpp
    src/rasterizer_test.cpp
    src/dilation_test.cpp
    src/material_test.cpp
    src/atlas_test.cpp
    src/light_test.cpp
)

target_link_libraries(sh_baker_test PRIVATE
    sh_baker
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sh_baker_test)

# --- Visualizer ---

add_executable(visualizer src/visualizer_main.cpp)


target_link_libraries(visualizer PRIVATE
    sh_baker
    glfw
    OpenGL::GL
)

# --- Converter Tool ---

add_executable(sh_cvt src/sh_cvt_main.cpp)
target_link_libraries(sh_cvt PRIVATE
    sh_baker
    gflags
    glog::glog
)
