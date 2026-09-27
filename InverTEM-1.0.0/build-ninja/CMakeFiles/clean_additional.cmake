# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "CMakeFiles\\InverTEM_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\InverTEM_autogen.dir\\ParseCache.txt"
  "CMakeFiles\\invertem-native-tests_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\invertem-native-tests_autogen.dir\\ParseCache.txt"
  "InverTEM_autogen"
  "invertem-native-tests_autogen"
  )
endif()
