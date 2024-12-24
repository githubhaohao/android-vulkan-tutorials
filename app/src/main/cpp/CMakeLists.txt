# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(jnilibs "${CMAKE_SOURCE_DIR}/../jniLibs")

set(VK_WRAPPER_DIR ${CMAKE_SOURCE_DIR}/wrapper)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DVK_USE_PLATFORM_ANDROID_KHR -DVK_NO_PROTOTYPES")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
file(GLOB src-files
        ${CMAKE_SOURCE_DIR}/*.cpp
        ${CMAKE_SOURCE_DIR}/render/VkRenderContext.cpp
        ${CMAKE_SOURCE_DIR}/util/*.cpp
        ${CMAKE_SOURCE_DIR}/examples/*.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample05_PushConstants.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample10_SpecializationInfo.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample11_CubeMap.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample12_InputAttachments.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample13_OffScreenRendering.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample14_DepthTesting.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample15_StencilTesting.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample16_MultiSampling.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample17_MultiThreading.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample18_Instancing.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample19_ReadPixels.cpp
        ${CMAKE_SOURCE_DIR}/examples/VkSample20_ComputeShader.cpp

)

add_library(native-app-glue STATIC ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_subdirectory(base ${CMAKE_SOURCE_DIR}/base)

include_directories(${VK_WRAPPER_DIR})
include_directories(${EXTERNAL_DIR})
include_directories(${EXTERNAL_DIR}/glm)
include_directories(${EXTERNAL_DIR}/imgui)
include_directories(${ANDROID_NDK}/sources/android/native_app_glue)

include_directories(
        util
        render
        examples
        shaderc/include
        wrapper
)

add_library( # Sets the name of the library.
             native-render

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${src-files}
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

target_link_libraries( # Specifies the target library.
                       native-render

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                        native-app-glue
                        libbase
                        android
                        z
)
