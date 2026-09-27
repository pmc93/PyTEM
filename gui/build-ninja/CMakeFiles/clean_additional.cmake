# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "CMakeFiles\\pytem-inversion-gui_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\pytem-inversion-gui_autogen.dir\\ParseCache.txt"
  "pytem-inversion-gui_autogen"
  )
endif()
