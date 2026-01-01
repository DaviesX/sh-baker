cmake_minimum_required(VERSION 3.14)
project(sh_baker CXX)

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
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# Prevent installing gtest/gmock
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

# Embree (Expected to be installed on system)
find_package(embree 3 REQUIRED)

# Eigen3 (Expected to be installed on system)
find_package(Eigen3 REQUIRED)

# --- Sources ---

set(SOURCES
    src/main.cpp
    src/io.cpp
    src/index.cpp
    src/occlusion.cpp
)

set(HEADERS
    src/scene.h
    src/io.h
    src/index.h
    src/occlusion.h
)

# --- Executable ---

add_executable(sh_baker ${SOURCES} ${HEADERS})

target_include_directories(sh_baker PRIVATE 
    src 
    ${tinygltf_SOURCE_DIR}
)

target_link_libraries(sh_baker PRIVATE
    tinygltf
    glog::glog
    gflags::gflags
    embree
    Eigen3::Eigen
)

# --- Tests ---

enable_testing()

add_executable(sh_baker_test
    src/io_test.cpp
    src/io.cpp
    src/index_test.cpp
    src/index.cpp
    src/occlusion_test.cpp
    src/occlusion.cpp
)

target_include_directories(sh_baker_test PRIVATE 
    src 
    ${tinygltf_SOURCE_DIR}
)

target_link_libraries(sh_baker_test PRIVATE
    gtest_main
    tinygltf
    glog::glog
    gflags::gflags
    embree
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(sh_baker_test)
