cmake_minimum_required(VERSION 3.14)
project(sh_baker CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Enable offline build support:
# 1. Store dependencies in a persistent 'deps' folder within the source tree.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/deps")
# 2. Prevent FetchContent from attempting to connect to the internet to check for updates.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# --- Dependencies ---

# tinygltf
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG        release
)
FetchContent_MakeAvailable(tinygltf)

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
FetchContent_Declare(
  stb_image
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb_image)

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

# --- Library (sh_baker_lib) ---

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

add_library(sh_baker_lib SHARED ${LIB_SOURCES} ${LIB_HEADERS})

target_include_directories(sh_baker_lib PUBLIC 
    src 
    ${tinygltf_SOURCE_DIR}
    ${stb_image_SOURCE_DIR}
    ${xatlas_SOURCE_DIR}/source/xatlas
    ${mikktspace_SOURCE_DIR}
)

target_link_libraries(sh_baker_lib PUBLIC
    tinygltf
    glog::glog
    gflags
    embree
    Eigen3::Eigen
    TBB::tbb
)

# --- Executable (sh_baker) ---

add_executable(sh_baker src/main.cpp)
target_link_libraries(sh_baker PRIVATE sh_baker_lib)

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
    sh_baker_lib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sh_baker_test)

# --- Visualizer ---

add_executable(visualizer src/visualizer_main.cpp)


target_link_libraries(visualizer PRIVATE
    sh_baker_lib
    glfw
    OpenGL::GL
)

# --- Converter Tool ---

add_executable(sh_cvt src/sh_cvt_main.cpp)
target_link_libraries(sh_cvt PRIVATE
    sh_baker_lib
    gflags
    glog::glog
)
