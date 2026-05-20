# NOTE: 이 프로젝트는 ament_python 빌드 타입을 사용합니다 (package.xml 참조).
# colcon build 시 실제로는 setup.py가 사용되며, 이 CMakeLists.txt는 참고용입니다.
cmake_minimum_required(VERSION 3.8)
project(camera_system)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install model files
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

