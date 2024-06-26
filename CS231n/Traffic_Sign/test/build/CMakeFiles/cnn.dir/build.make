# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build

# Include any dependencies generated for this target.
include CMakeFiles/cnn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnn.dir/flags.make

CMakeFiles/cnn.dir/cnn.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cnn.cpp.o: ../cnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnn.dir/cnn.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cnn.dir/cnn.cpp.o -c /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/cnn.cpp

CMakeFiles/cnn.dir/cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cnn.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/cnn.cpp > CMakeFiles/cnn.dir/cnn.cpp.i

CMakeFiles/cnn.dir/cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cnn.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/cnn.cpp -o CMakeFiles/cnn.dir/cnn.cpp.s

CMakeFiles/cnn.dir/cnn.cpp.o.requires:

.PHONY : CMakeFiles/cnn.dir/cnn.cpp.o.requires

CMakeFiles/cnn.dir/cnn.cpp.o.provides: CMakeFiles/cnn.dir/cnn.cpp.o.requires
	$(MAKE) -f CMakeFiles/cnn.dir/build.make CMakeFiles/cnn.dir/cnn.cpp.o.provides.build
.PHONY : CMakeFiles/cnn.dir/cnn.cpp.o.provides

CMakeFiles/cnn.dir/cnn.cpp.o.provides.build: CMakeFiles/cnn.dir/cnn.cpp.o


# Object files for target cnn
cnn_OBJECTS = \
"CMakeFiles/cnn.dir/cnn.cpp.o"

# External object files for target cnn
cnn_EXTERNAL_OBJECTS =

cnn: CMakeFiles/cnn.dir/cnn.cpp.o
cnn: CMakeFiles/cnn.dir/build.make
cnn: /usr/local/lib/libopencv_stitching.so.3.2.0
cnn: /usr/local/lib/libopencv_superres.so.3.2.0
cnn: /usr/local/lib/libopencv_videostab.so.3.2.0
cnn: /usr/local/lib/libopencv_aruco.so.3.2.0
cnn: /usr/local/lib/libopencv_bgsegm.so.3.2.0
cnn: /usr/local/lib/libopencv_bioinspired.so.3.2.0
cnn: /usr/local/lib/libopencv_ccalib.so.3.2.0
cnn: /usr/local/lib/libopencv_dpm.so.3.2.0
cnn: /usr/local/lib/libopencv_freetype.so.3.2.0
cnn: /usr/local/lib/libopencv_fuzzy.so.3.2.0
cnn: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
cnn: /usr/local/lib/libopencv_optflow.so.3.2.0
cnn: /usr/local/lib/libopencv_reg.so.3.2.0
cnn: /usr/local/lib/libopencv_saliency.so.3.2.0
cnn: /usr/local/lib/libopencv_stereo.so.3.2.0
cnn: /usr/local/lib/libopencv_structured_light.so.3.2.0
cnn: /usr/local/lib/libopencv_surface_matching.so.3.2.0
cnn: /usr/local/lib/libopencv_tracking.so.3.2.0
cnn: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
cnn: /usr/local/lib/libopencv_ximgproc.so.3.2.0
cnn: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
cnn: /usr/local/lib/libopencv_xphoto.so.3.2.0
cnn: /usr/local/lib/libopencv_shape.so.3.2.0
cnn: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
cnn: /usr/local/lib/libopencv_rgbd.so.3.2.0
cnn: /usr/local/lib/libopencv_calib3d.so.3.2.0
cnn: /usr/local/lib/libopencv_video.so.3.2.0
cnn: /usr/local/lib/libopencv_datasets.so.3.2.0
cnn: /usr/local/lib/libopencv_dnn.so.3.2.0
cnn: /usr/local/lib/libopencv_face.so.3.2.0
cnn: /usr/local/lib/libopencv_plot.so.3.2.0
cnn: /usr/local/lib/libopencv_text.so.3.2.0
cnn: /usr/local/lib/libopencv_features2d.so.3.2.0
cnn: /usr/local/lib/libopencv_flann.so.3.2.0
cnn: /usr/local/lib/libopencv_objdetect.so.3.2.0
cnn: /usr/local/lib/libopencv_ml.so.3.2.0
cnn: /usr/local/lib/libopencv_highgui.so.3.2.0
cnn: /usr/local/lib/libopencv_photo.so.3.2.0
cnn: /usr/local/lib/libopencv_videoio.so.3.2.0
cnn: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
cnn: /usr/local/lib/libopencv_imgproc.so.3.2.0
cnn: /usr/local/lib/libopencv_core.so.3.2.0
cnn: CMakeFiles/cnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnn.dir/build: cnn

.PHONY : CMakeFiles/cnn.dir/build

CMakeFiles/cnn.dir/requires: CMakeFiles/cnn.dir/cnn.cpp.o.requires

.PHONY : CMakeFiles/cnn.dir/requires

CMakeFiles/cnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnn.dir/clean

CMakeFiles/cnn.dir/depend:
	cd /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/test/build/CMakeFiles/cnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cnn.dir/depend

