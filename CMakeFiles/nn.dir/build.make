# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/borys/Work/NeuralNetworks/nn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/borys/Work/NeuralNetworks/nn

# Include any dependencies generated for this target.
include CMakeFiles/nn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nn.dir/flags.make

CMakeFiles/nn.dir/src/main.cpp.o: CMakeFiles/nn.dir/flags.make
CMakeFiles/nn.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/borys/Work/NeuralNetworks/nn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nn.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn.dir/src/main.cpp.o -c /home/borys/Work/NeuralNetworks/nn/src/main.cpp

CMakeFiles/nn.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/borys/Work/NeuralNetworks/nn/src/main.cpp > CMakeFiles/nn.dir/src/main.cpp.i

CMakeFiles/nn.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/borys/Work/NeuralNetworks/nn/src/main.cpp -o CMakeFiles/nn.dir/src/main.cpp.s

CMakeFiles/nn.dir/src/nn.cpp.o: CMakeFiles/nn.dir/flags.make
CMakeFiles/nn.dir/src/nn.cpp.o: src/nn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/borys/Work/NeuralNetworks/nn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/nn.dir/src/nn.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn.dir/src/nn.cpp.o -c /home/borys/Work/NeuralNetworks/nn/src/nn.cpp

CMakeFiles/nn.dir/src/nn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn.dir/src/nn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/borys/Work/NeuralNetworks/nn/src/nn.cpp > CMakeFiles/nn.dir/src/nn.cpp.i

CMakeFiles/nn.dir/src/nn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn.dir/src/nn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/borys/Work/NeuralNetworks/nn/src/nn.cpp -o CMakeFiles/nn.dir/src/nn.cpp.s

# Object files for target nn
nn_OBJECTS = \
"CMakeFiles/nn.dir/src/main.cpp.o" \
"CMakeFiles/nn.dir/src/nn.cpp.o"

# External object files for target nn
nn_EXTERNAL_OBJECTS =

nn: CMakeFiles/nn.dir/src/main.cpp.o
nn: CMakeFiles/nn.dir/src/nn.cpp.o
nn: CMakeFiles/nn.dir/build.make
nn: CMakeFiles/nn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/borys/Work/NeuralNetworks/nn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable nn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nn.dir/build: nn

.PHONY : CMakeFiles/nn.dir/build

CMakeFiles/nn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nn.dir/clean

CMakeFiles/nn.dir/depend:
	cd /home/borys/Work/NeuralNetworks/nn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/borys/Work/NeuralNetworks/nn /home/borys/Work/NeuralNetworks/nn /home/borys/Work/NeuralNetworks/nn /home/borys/Work/NeuralNetworks/nn /home/borys/Work/NeuralNetworks/nn/CMakeFiles/nn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nn.dir/depend
