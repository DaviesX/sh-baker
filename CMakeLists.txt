cmake_minimum_required(VERSION 3.14)
project(sh_baker CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Dependencies ---

# tinygltf
FetchContent_Declare(
  tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG        release
)
FetchContent_MakeAvailable(tinygltf)

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

# Embree (Expected to be installed on system)
find_package(embree 4 REQUIRED)

# Eigen3 (Expected to be installed on system)
find_package(Eigen3 REQUIRED)

# --- Sources ---

set(SOURCES
    src/main.cpp
    src/loader.cpp
    src/scene.cpp
    src/occlusion.cpp
    src/sh_coeffs.cpp
    src/saver.cpp
    src/baker.cpp
    src/rasterizer.cpp
    src/tinyexr_impl.cpp
    src/miniz.c
)

set(HEADERS
    src/scene.h
    src/loader.h
    src/occlusion.h
    src/sh_coeffs.h
    src/saver.h
    src/baker.h
    src/rasterizer.h
)

# --- Executable ---

add_executable(sh_baker ${SOURCES} ${HEADERS})

target_include_directories(sh_baker PRIVATE 
    src 
    ${tinygltf_SOURCE_DIR}
    ${stb_image_SOURCE_DIR}
)

target_link_libraries(sh_baker PRIVATE
    tinygltf
    glog::glog
    gflags
    embree
    Eigen3::Eigen
)

# --- Tests ---

enable_testing()

add_executable(sh_baker_test
    src/loader_test.cpp
    src/loader.cpp
    src/scene_test.cpp
    src/scene.cpp
    src/occlusion_test.cpp
    src/occlusion.cpp
    src/sh_coeffs_test.cpp
    src/sh_coeffs.cpp
    src/saver_test.cpp
    src/saver.cpp
    src/tinyexr_impl.cpp
    src/miniz.c
    src/baker_test.cpp
    src/baker.cpp
    src/rasterizer_test.cpp
    src/rasterizer.cpp
)

target_include_directories(sh_baker_test PRIVATE 
    src 
    ${tinygltf_SOURCE_DIR}
    ${stb_image_SOURCE_DIR}
)

target_link_libraries(sh_baker_test PRIVATE
    GTest::gtest_main
    tinygltf
    glog::glog
    gflags
    embree
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(sh_baker_test)
