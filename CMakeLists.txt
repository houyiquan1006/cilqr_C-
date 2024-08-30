cmake_minimum_required(VERSION 3.0)
project(postproc)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen3 library
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 3.6 REQUIRED COMPONENTS Development NumPy)
find_package(OpenCV REQUIRED)

# Add sources
file(GLOB SOURCES 
    "src/*.cpp" 
    "src/cilqr_joint_planner/*.cpp"
    "src/cilqr_optimizer/*.cpp"
    # "src/common_utils/*.cpp"
    # "src/test/*.cpp"
    "src/optimize/*.cpp"
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
# Define the executable
add_executable(${PROJECT_NAME} 
    ${SOURCES}
)

# Include directories using target_include_directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/optimize
    ${PROJECT_SOURCE_DIR}/src/optimize/speedoptimize
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    Python3::Python
    Python3::Module
    Python3::NumPy
    Eigen3::Eigen
    stdc++fs
    ${PROJECT_SOURCE_DIR}/lib/libosqp.so
    ${OpenCV_LIBS}
)