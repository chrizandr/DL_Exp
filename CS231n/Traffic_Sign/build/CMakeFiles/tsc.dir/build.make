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
CMAKE_SOURCE_DIR = /home/chrizandr/DL_Exp/CS231n/Traffic_Sign

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build

# Include any dependencies generated for this target.
include CMakeFiles/tsc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsc.dir/flags.make

CMakeFiles/tsc.dir/tsc.cpp.o: CMakeFiles/tsc.dir/flags.make
CMakeFiles/tsc.dir/tsc.cpp.o: ../tsc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsc.dir/tsc.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsc.dir/tsc.cpp.o -c /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/tsc.cpp

CMakeFiles/tsc.dir/tsc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsc.dir/tsc.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/tsc.cpp > CMakeFiles/tsc.dir/tsc.cpp.i

CMakeFiles/tsc.dir/tsc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsc.dir/tsc.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/tsc.cpp -o CMakeFiles/tsc.dir/tsc.cpp.s

CMakeFiles/tsc.dir/tsc.cpp.o.requires:

.PHONY : CMakeFiles/tsc.dir/tsc.cpp.o.requires

CMakeFiles/tsc.dir/tsc.cpp.o.provides: CMakeFiles/tsc.dir/tsc.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsc.dir/build.make CMakeFiles/tsc.dir/tsc.cpp.o.provides.build
.PHONY : CMakeFiles/tsc.dir/tsc.cpp.o.provides

CMakeFiles/tsc.dir/tsc.cpp.o.provides.build: CMakeFiles/tsc.dir/tsc.cpp.o


# Object files for target tsc
tsc_OBJECTS = \
"CMakeFiles/tsc.dir/tsc.cpp.o"

# External object files for target tsc
tsc_EXTERNAL_OBJECTS =

tsc: CMakeFiles/tsc.dir/tsc.cpp.o
tsc: CMakeFiles/tsc.dir/build.make
tsc: /usr/local/lib/libopencv_stitching.so.3.2.0
tsc: /usr/local/lib/libopencv_superres.so.3.2.0
tsc: /usr/local/lib/libopencv_videostab.so.3.2.0
tsc: /usr/local/lib/libopencv_aruco.so.3.2.0
tsc: /usr/local/lib/libopencv_bgsegm.so.3.2.0
tsc: /usr/local/lib/libopencv_bioinspired.so.3.2.0
tsc: /usr/local/lib/libopencv_ccalib.so.3.2.0
tsc: /usr/local/lib/libopencv_dpm.so.3.2.0
tsc: /usr/local/lib/libopencv_freetype.so.3.2.0
tsc: /usr/local/lib/libopencv_fuzzy.so.3.2.0
tsc: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
tsc: /usr/local/lib/libopencv_optflow.so.3.2.0
tsc: /usr/local/lib/libopencv_reg.so.3.2.0
tsc: /usr/local/lib/libopencv_saliency.so.3.2.0
tsc: /usr/local/lib/libopencv_stereo.so.3.2.0
tsc: /usr/local/lib/libopencv_structured_light.so.3.2.0
tsc: /usr/local/lib/libopencv_surface_matching.so.3.2.0
tsc: /usr/local/lib/libopencv_tracking.so.3.2.0
tsc: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
tsc: /usr/local/lib/libopencv_ximgproc.so.3.2.0
tsc: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
tsc: /usr/local/lib/libopencv_xphoto.so.3.2.0
tsc: /usr/local/lib/libopencv_shape.so.3.2.0
tsc: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
tsc: /usr/local/lib/libopencv_rgbd.so.3.2.0
tsc: /usr/local/lib/libopencv_calib3d.so.3.2.0
tsc: /usr/local/lib/libopencv_video.so.3.2.0
tsc: /usr/local/lib/libopencv_datasets.so.3.2.0
tsc: /usr/local/lib/libopencv_dnn.so.3.2.0
tsc: /usr/local/lib/libopencv_face.so.3.2.0
tsc: /usr/local/lib/libopencv_plot.so.3.2.0
tsc: /usr/local/lib/libopencv_text.so.3.2.0
tsc: /usr/local/lib/libopencv_features2d.so.3.2.0
tsc: /usr/local/lib/libopencv_flann.so.3.2.0
tsc: /usr/local/lib/libopencv_objdetect.so.3.2.0
tsc: /usr/local/lib/libopencv_ml.so.3.2.0
tsc: /usr/local/lib/libopencv_highgui.so.3.2.0
tsc: /usr/local/lib/libopencv_photo.so.3.2.0
tsc: /usr/local/lib/libopencv_videoio.so.3.2.0
tsc: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
tsc: /usr/local/lib/libopencv_imgproc.so.3.2.0
tsc: /usr/local/lib/libopencv_core.so.3.2.0
tsc: CMakeFiles/tsc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsc.dir/build: tsc

.PHONY : CMakeFiles/tsc.dir/build

CMakeFiles/tsc.dir/requires: CMakeFiles/tsc.dir/tsc.cpp.o.requires

.PHONY : CMakeFiles/tsc.dir/requires

CMakeFiles/tsc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsc.dir/clean

CMakeFiles/tsc.dir/depend:
	cd /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chrizandr/DL_Exp/CS231n/Traffic_Sign /home/chrizandr/DL_Exp/CS231n/Traffic_Sign /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build /home/chrizandr/DL_Exp/CS231n/Traffic_Sign/build/CMakeFiles/tsc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsc.dir/depend

