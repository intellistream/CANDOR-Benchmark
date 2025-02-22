#set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
#set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
cmake_minimum_required(VERSION 3.27)
project(CANDYBENCH CXX)

option(ENABLE_CUDA "Enable cuda" ON)
message(STATUS "1.0 CUDA enabled: ${ENABLE_CUDA}")
include (cmake/FindCuda.cmake)
include (cmake/FindTorch.cmake)
#set(CMAKE_CUDA_ARCHITECTURES "70;75;80")
set(CMAKE_CUDA_ARCHITECTURES ALL)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)




include_directories (${gflags_INCLUDE_DIR})
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_RULE_MESSAGES OFF)
# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/default.cmake)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#gcc 10 g++10
# First compile faiss before anything else
#set(CMAKE_CXX_FLAGS "-fno-openmp")
#test avx2

# Option to enable/disable CUDA
message(STATUS "2.0 CUDA enabled: ${ENABLE_CUDA}")
if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    add_definitions(-DENABLE_CUDA=1)
    message(STATUS "CUDA is enabled")
else()
    message(STATUS "CUDA is not enabled")
endif ()

add_subdirectory(thirdparty/faiss)

# Set specific options for Faiss compilation
#target_compile_options(faiss PRIVATE "-fno-openmp")
set(LIBRARIES ${LIBRARIES} faiss)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBRARIES ${LIBRARIES} ${TORCH_LIBRARIES})


# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Wno-interference-size")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DCANDY_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")
set(PROJECT_BINARY_DIR_RAW ${PROJECT_BINARY_DIR})

# Valid values are "generic", "avx2", "avx512".

detect_avx512_support(AVX512_AVAILABLE)
# Use AVX-512 based on the result
if(AVX512_AVAILABLE)
    message(STATUS "AVX-512 support detected.")
    set(CANDY_AVX512 1)
    set(CANDY_AVX2 1)
else()
    message(STATUS "AVX-512 support NOT detected.")
    detect_avx2_support(AVX2_AVAILABLE)
    if(AVX2_AVAILABLE)
        message(STATUS "AVX-2 support detected.")
        set(CANDY_AVX2 1)
    else ()
        message(STATUS "AVX-2 support not detected.")
        set(CANDY_AVX2 0)
        set(CANDY_AVX512 0)
    endif ()
endif()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/simd_config.h.in"
        "${PROJECT_BINARY_DIR}/include/simd_config.h"
)


#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_FLAGS "-std=c++11")

option(ENABLE_OPENCL
        "Enable opencl support"
        OFF
)
option(ENABLE_RAY
        "Enable the distributed computing powered by ray"
        OFF
)
option(ENABLE_SPTAG
        "Integrate SPTAG LIBS"
        OFF
)
option(ENABLE_PUCK
        "Integrate PUCK LIBS from https://github.com/baidu/puck"
        OFF
)

option(ENABLE_PYBIND
        "Enable original pybind and build CANDYBENCH python"
        OFF
)
option(ENABLE_DiskANN
        "Enable DiskANN support"
        OFF
)

if (NOT MSVC)
    set(DISKANN_ASYNC_LIB aio)
endif()

set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
#	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra")
if (NOT PYBIND)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
else()
    #-Ofast is super problematic for python.  see: https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -march=native -mtune=native -ftree-vectorize")
    add_compile_options(-fPIC)
endif()
add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2)

#OPTIONAL OPENCL
if (NOT ENABLE_OPENCL)
    message(STATUS "I will NOT include opencl support ")
    set(CANDY_CL 0)
else ()
    set(CANDY_CL 1)
    message(STATUS "I will include opencl support, pls make sure you have corresponding libs")
    set(LIBRARIES ${LIBRARIES} OpenCL)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/opencl_config.h.in"
        "${PROJECT_BINARY_DIR}/include/opencl_config.h"
)
option(ENABLE_PAPI
        "Enable papi support, pls first compile papi or set REBUILD_PAPI to ON"
        OFF
)
# OPTIONAL PAPI
if (NOT ENABLE_PAPI)
    message(STATUS "I will NOT use PAPI ")
    set(CANDY_PAPI 0)
else ()
    set(CANDY_PAPI 1)
    message(STATUS "I will try to use PAPI for HW counters, pls make sure your arch supports it")
    option(REBUILD_PAPI
            "Rebuild the papi lib"
            OFF
    )
    if (REBUILD_PAPI)
        set(PAPISCRIPTPATH ../thirdparty)
        execute_process(COMMAND bash ${PAPISCRIPTPATH}/makeClean.sh WORKING_DIRECTORY ${PAPISCRIPTPATH})
        execute_process(COMMAND bash ${PAPISCRIPTPATH}/installPAPI.sh WORKING_DIRECTORY ${PAPISCRIPTPATH})
        message(STATUS "I have rebuilt PAPI")
    endif ()
    find_library(libPAPI libpapi.so ./thirdparty/papi_build/lib)
    message(STATUS "papiLibs= " ${libPAPI})
    set(LIBRARIES ${LIBRARIES} ${libPAPI})

endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/papi_config.h.in"
        "${PROJECT_BINARY_DIR}/include/papi_config.h"
)
#options of ray
if (NOT ENABLE_RAY)
    message(STATUS "I will NOT include ray or distributed features ")
    set(CANDY_RAY 0)
else ()
    set(CANDY_RAY 1)
    message(STATUS "I will include ray and distributed computing support, pls make sure you have corresponding libs")
    if (NOT DEFINED ENV{RAYPATH})
        set(ENV{RAYPATH} '/usr/lib')
        message(STATUS "auto set RAYPATH= " $ENV{RAYPATH})
    endif ()
    set(RAY_DIR $ENV{RAYPATH})
    set(LIBRARIES ${LIBRARIES} ${RAY_DIR}/libray_api.so)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rayHeader)
endif ()
set(CANDY_HDF5 0)
configure_file(
        "${PROJECT_SOURCE_DIR}/include/ray_config.h.in"
        "${PROJECT_BINARY_DIR}/include/ray_config.h"
)
#set(CMAKE_CUDA_ARCHITECTURES 55)
# Set LOGGING_LEVEL Flag
#option of hdf5

configure_file(
        "${PROJECT_SOURCE_DIR}/include/hdf5_config.h.in"
        "${PROJECT_BINARY_DIR}/include/hdf5_config.h"
)

# Set SPTAG build directory
set(SPTAG_BUILD_DIR ${CMAKE_BINARY_DIR}/SPTAG_build)

# Configure build output directories specifically for SPTAG
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SPTAG_BUILD_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SPTAG_BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${SPTAG_BUILD_DIR}/lib)
if (NOT ENABLE_SPTAG)
    message(STATUS "I will NOT include SPTAG LIBS")
    set(CANDY_SPTAG 0)
else ()
    set(CANDY_SPTAG 1)
    message(STATUS "I will include support for SPTAG")
    add_subdirectory(thirdparty/SPTAG)
    set(LIBRARIES ${LIBRARIES}  SPTAGLib)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/sptag_config.h.in"
        "${PROJECT_BINARY_DIR}/include/sptag_config.h"
)


set(DiskANN_BUILD_DIR ${CMAKE_BINARY_DIR}/DiskANN_build)
# Configure build output directories specifically for DISKANN
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DiskANN_BUILD_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DiskANN_BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DiskANN_BUILD_DIR}/lib)

if (NOT ENABLE_DiskANN)
    message(STATUS "I will NOT include diskann LIBS")
    set(CANDY_DiskANN 0)
else ()
    set(CANDY_DiskANN 1)
    file(GLOB_RECURSE DISKANN_SRC_FILES
            "thirdparty/DiskANN/src/*.cpp"
            "thirdparty/DiskANN/python/src/*.cpp"
    )
    file(GLOB DISKANN_SRC_FILES "thirdparty/DiskANN/src/*.cpp"  )
    file(GLOB DISKANN_HEADER_FILES "thirdparty/DiskANN/include/*.h")
    message(STATUS "I will include support for DiskANN")

    # Add DiskANN
    find_package(OpenMP REQUIRED)


    add_subdirectory(thirdparty/DiskANN)
    include_directories(thirdparty/DiskANN/include)
    #    add_sources(
    #            thirdparty/DiskANN/src/pq_flash_index.cpp
    #
    #    )
    set(LIBRARIES ${LIBRARIES}  diskann)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/diskann_config.h.in"
        "${PROJECT_BINARY_DIR}/include/diskann_config.h"
)

# Set PUCK build directory
set(PUCK_BUILD_DIR ${CMAKE_BINARY_DIR}/PUCK_build)

# Configure build output directories specifically for PUCK
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PUCK_BUILD_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PUCK_BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PUCK_BUILD_DIR}/lib)
if (NOT ENABLE_PUCK)
    message(STATUS "I will NOT include PUCK LIBS")
    set(CANDY_PUCK 0)
else ()
    set(CANDY_PUCK 1)

    message(STATUS "I will include support for PUCK")
    set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/gflags" ${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/glog" ${CMAKE_PREFIX_PATH})
    find_package(gflags)
    find_package(glog)
    #set(GFLAGS_LIBRARIES /usr/local/lib/libgflags.so)
    #set(GLOG_LIBRARIES /usr/local/lib/libglog.so)
    #
    add_compile_definitions(GLOG_USE_GLOG_EXPORT)
    set(LIBRARIES ${LIBRARIES} gflags)
    #
    set(LIBRARIES ${LIBRARIES} glog)



    add_subdirectory(thirdparty/puck)

    set(LIBRARIES ${LIBRARIES}  puck)


endif ()

message(STATUS " Libraries: ${LIBRARIES}")
configure_file(
        "${PROJECT_SOURCE_DIR}/include/puck_config.h.in"
        "${PROJECT_BINARY_DIR}/include/puck_config.h"
)

# Reset output directories for the rest of your project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
#pytorch
#set(Torch_DIR "/home/tony/.local/lib/python3.10/site-packages/torch/share/cmake" )
# Log4cc
#find_package(Log4cxx REQUIRED)
#include_directories(${Log4cxx_INCLUDE_DIR})
#set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})


option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")


# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach ()


#add_subdirectory(pytorchNN)
# Add Source Code


add_subdirectory(src)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

# Add Library
get_sources(CANDY_SOURCE_FILES)
get_headers(CANDY_HEADER_FILES)

if (ENABLE_CUDA)
    set_source_files_properties(src/CANDY/IndexTable.cpp
            PROPERTIES
            LANGUAGE CUDA
    )
endif ()

add_library(CANDYBENCH SHARED ${CANDY_SOURCE_FILES} ${CANDY_HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR}
        src/CANDY/IndexTable.cpp)

set_property(TARGET CANDYBENCH PROPERTY CXX_STANDARD 20)
target_include_directories(CANDYBENCH PUBLIC "include")

if (ENABLE_CUDA)
    set(LIBRARIES ${LIBRARIES} cublas cudart)
    set_target_properties(CANDYBENCH PROPERTIES
            CUDA_STANDARD 20
            CXX_STANDARD 20
    )
else ()
    set_target_properties(CANDYBENCH PROPERTIES
            CXX_STANDARD 20
    )
endif ()

# 设置 MKL 库的路径
set(MKL_INCLUDE_DIR "/usr/include/mkl")
set(MKL_LIB_DIR "/usr/lib/x86_64-linux-gnu")
#set(MPI_INCLUDE_PATH "/usr/include/openmpi-x86_64")
#set(MPI_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")

set(MKL_LIBRARIES
        "${MKL_LIB_DIR}/libmkl_intel_lp64.so"
        "${MKL_LIB_DIR}/libmkl_sequential.so"
        "${MKL_LIB_DIR}/libmkl_core.so"
)
target_include_directories(CANDYBENCH PUBLIC ${MKL_INCLUDE_DIR})
# MKL 和其他库的链接
target_link_libraries(CANDYBENCH PUBLIC
        ${MKL_LIBRARIES}
        iomp5
        pthread
        m
        dl
)
target_compile_options(CANDYBENCH PUBLIC
        -m64
        -DUSE_MKL
        -I${MKL_INCLUDE_DIR}
)
link_directories(${MKL_LIB_DIR})

#these 2 ugly lines are for puck
#add_definitions(-DFINTEGER=int)
#target_include_directories(CANDYBENCH PUBLIC "/usr/include/mkl")

target_include_directories(CANDYBENCH PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(CANDYBENCH PUBLIC "thirdparty/papi_build/include")
target_include_directories(CANDYBENCH PUBLIC "thirdparty/")
if (ENABLE_PUCK)
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/puck")
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/puck/puck")
endif ()
if (ENABLE_SPTAG)
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/SPTAG")
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/SPTAG/AnnService")
endif ()
if (ENABLE_DiskANN)
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/DiskANN")
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/DiskANN/python/include")
    target_include_directories(CANDYBENCH PUBLIC "thirdparty/DiskANN/include")
    target_sources(CANDYBENCH PUBLIC ${DISKANN_SRC_FILES})
    target_sources(CANDYBENCH PUBLIC thirdparty/DiskANN/python/src/dynamic_memory_index.cpp)
endif ()
if(OpenMP_CXX_FOUND)
    target_link_libraries(CANDYBENCH PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_options(CANDYBENCH PUBLIC "-Wl,-rpath,./")
target_link_options(CANDYBENCH PUBLIC "-Wl,-rpath,./")
message(WARNING "CANDY GFLAGS_LIBRARY_DIR: ${GFLAGS_LIBRARIES}")
message(WARNING "CANDY GLOG_LIBRARY_DIR: ${GLOG_LIBRARIES}")
#target_link_libraries(CANDYBENCH PUBLIC  ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
target_link_libraries(CANDYBENCH PUBLIC ${LIBRARIES})
install(TARGETS CANDYBENCH LIBRARY DESTINATION .)
find_library(LIBAIO_LIBRARIES NAMES aio)
message(STATUS "LIBAIO_LIBRARIES: ${LIBAIO_LIBRARIES}")
target_link_libraries(CANDYBENCH PUBLIC ${LIBAIO_LIBRARIES})
#add_subdirectory(test)
#Add benchmarks with command
add_subdirectory(benchmark)
#Add tests (based on GTest suits)
#include(GoogleTest)

if (ENABLE_UNIT_TESTS)
    #add_subdirectory(test)
endif ()

if (NOT ENABLE_PYBIND)
    message(STATUS "I will NOT build original python package PyCANDYAlgo")
    set(CANDY_PYBIND 0)
else ()
    message(STATUS "I will  build original python package PyCANDYAlgo")
    add_subdirectory(thirdparty/pybind11)
    pybind11_add_module(PyCANDYAlgo ${PROJECT_SOURCE_DIR}/src/PyCANDY.cpp)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
    #target_link_libraries(PyCANDYAlgo PRIVATE gflags glog)
    target_link_libraries(PyCANDYAlgo PUBLIC ${LIBRARIES} CANDYBENCH ${TORCH_PYTHON_LIBRARY})
    install(TARGETS PyCANDYAlgo LIBRARY DESTINATION .)
    message(STATUS "Additional include torch_python: ${TORCH_PYTHON_LIBRARY}")
    set(CANDY_PYBIND 1)
    set_property(TARGET PyCANDYAlgo  PROPERTY CXX_STANDARD 20)
endif ()
configure_file(
        "${PROJECT_SOURCE_DIR}/include/pybind_config.h.in"
        "${PROJECT_BINARY_DIR}/include/pybind_config.h"
)
install(DIRECTORY "include" DESTINATION "/CANDY" COMPONENT CANDYBENCH)
# copy scripts
file(GLOB allCopyFiles "${PROJECT_SOURCE_DIR}/scripts/*")
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
file(COPY ${allCopyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/benchmark/scripts)
message(STATUS "build path:" ${CMAKE_CURRENT_BINARY_DIR})
# copy files needed for real world matrix loader
set(source_directory "${CMAKE_SOURCE_DIR}/benchmark/datasets/")
set(destination_directory "${CMAKE_BINARY_DIR}/benchmark/datasets/")
file(COPY ${source_directory} DESTINATION ${destination_directory})