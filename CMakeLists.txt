cmake_minimum_required(VERSION 3.9)

project(cholesky VERSION 0.0.1 DESCRIPTION "sparse cholesky")
include(GNUInstallDirs)

add_library(mmio SHARED mmio.c mnd.c)
set_target_properties(mmio PROPERTIES PUBLIC_HEADER "mmio.h mnd.h")
target_compile_options(mmio PRIVATE -I $ENV{LG_RT_DIR})
install(TARGETS mmio
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(cholesky SHARED cholesky.cc)
set_target_properties(cholesky PROPERTIES PUBLIC_HEADER cholesky.h)
target_compile_options(cholesky PRIVATE -I $ENV{LG_RT_DIR})
install(TARGETS mmio
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
