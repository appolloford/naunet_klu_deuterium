# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1005/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1005/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/cjhsu/naunet/naunet_klu_deuterium

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/cjhsu/naunet/naunet_klu_deuterium/build

# Include any dependencies generated for this target.
include src/CMakeFiles/naunet_shared.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/naunet_shared.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/naunet_shared.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/naunet_shared.dir/flags.make

src/CMakeFiles/naunet_shared.dir/naunet.cpp.o: src/CMakeFiles/naunet_shared.dir/flags.make
src/CMakeFiles/naunet_shared.dir/naunet.cpp.o: ../src/naunet.cpp
src/CMakeFiles/naunet_shared.dir/naunet.cpp.o: src/CMakeFiles/naunet_shared.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/cjhsu/naunet/naunet_klu_deuterium/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/naunet_shared.dir/naunet.cpp.o"
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/naunet_shared.dir/naunet.cpp.o -MF CMakeFiles/naunet_shared.dir/naunet.cpp.o.d -o CMakeFiles/naunet_shared.dir/naunet.cpp.o -c /scratch/cjhsu/naunet/naunet_klu_deuterium/src/naunet.cpp

src/CMakeFiles/naunet_shared.dir/naunet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/naunet_shared.dir/naunet.cpp.i"
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/cjhsu/naunet/naunet_klu_deuterium/src/naunet.cpp > CMakeFiles/naunet_shared.dir/naunet.cpp.i

src/CMakeFiles/naunet_shared.dir/naunet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/naunet_shared.dir/naunet.cpp.s"
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/cjhsu/naunet/naunet_klu_deuterium/src/naunet.cpp -o CMakeFiles/naunet_shared.dir/naunet.cpp.s

# Object files for target naunet_shared
naunet_shared_OBJECTS = \
"CMakeFiles/naunet_shared.dir/naunet.cpp.o"

# External object files for target naunet_shared
naunet_shared_EXTERNAL_OBJECTS = \
"/scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_constants.dir/naunet_constants.cpp.o" \
"/scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_physics.dir/naunet_physics.cpp.o" \
"/scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_rates.dir/naunet_rates.cpp.o" \
"/scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_fex.dir/naunet_fex.cpp.o" \
"/scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_jac.dir/naunet_jac.cpp.o"

src/libnaunet.so: src/CMakeFiles/naunet_shared.dir/naunet.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_constants.dir/naunet_constants.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_physics.dir/naunet_physics.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_rates.dir/naunet_rates.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_fex.dir/naunet_fex.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_jac.dir/naunet_jac.cpp.o
src/libnaunet.so: src/CMakeFiles/naunet_shared.dir/build.make
src/libnaunet.so: /usr/local/sundials-5.7.0/lib/libsundials_cvode.so.5.7.0
src/libnaunet.so: /usr/local/sundials-5.7.0/lib/libsundials_sunlinsolklu.so.3.7.0
src/libnaunet.so: /usr/local/sundials-5.7.0/lib/libsundials_sunmatrixsparse.so.3.7.0
src/libnaunet.so: /scratch/cjhsu/usr/suitesparse/SuiteSparse-5.9.0/lib/libklu.so
src/libnaunet.so: /scratch/cjhsu/usr/suitesparse/SuiteSparse-5.9.0/lib/libamd.so
src/libnaunet.so: /scratch/cjhsu/usr/suitesparse/SuiteSparse-5.9.0/lib/libcolamd.so
src/libnaunet.so: /scratch/cjhsu/usr/suitesparse/SuiteSparse-5.9.0/lib/libbtf.so
src/libnaunet.so: /scratch/cjhsu/usr/suitesparse/SuiteSparse-5.9.0/lib/libsuitesparseconfig.so
src/libnaunet.so: src/CMakeFiles/naunet_shared.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/cjhsu/naunet/naunet_klu_deuterium/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libnaunet.so"
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/naunet_shared.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/naunet_shared.dir/build: src/libnaunet.so
.PHONY : src/CMakeFiles/naunet_shared.dir/build

src/CMakeFiles/naunet_shared.dir/clean:
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src && $(CMAKE_COMMAND) -P CMakeFiles/naunet_shared.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/naunet_shared.dir/clean

src/CMakeFiles/naunet_shared.dir/depend:
	cd /scratch/cjhsu/naunet/naunet_klu_deuterium/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/cjhsu/naunet/naunet_klu_deuterium /scratch/cjhsu/naunet/naunet_klu_deuterium/src /scratch/cjhsu/naunet/naunet_klu_deuterium/build /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src /scratch/cjhsu/naunet/naunet_klu_deuterium/build/src/CMakeFiles/naunet_shared.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/naunet_shared.dir/depend
