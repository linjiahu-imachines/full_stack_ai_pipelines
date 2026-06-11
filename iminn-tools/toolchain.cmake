# Cross-compilation toolchain file with template variables
# @CROSS_ARCH@ - Target architecture (x86_64, aarch64, or riscv64)
# @CROSS_TRIPLE@ - Target triple (e.g., aarch64-linux-gnu)
# @CROSS_FLAGS@ - Additional compiler flags
# @CROSS_COMP@ - Compiler to use (gcc or clang)
# Determine if we're cross-compiling or doing native compilation

function(get_cross_var VAR_NAME)
    set(options REQUIRED)
    set(oneValueArgs DEFAULT)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})

    # Try to get from environment if not already defined
    if(NOT DEFINED ${VAR_NAME} AND DEFINED ENV{${VAR_NAME}})
        set(${VAR_NAME} "$ENV{${VAR_NAME}}" CACHE STRING "Set from environment variable ${VAR_NAME}")
    endif()

    # Use default if still undefined
    if(NOT DEFINED ${VAR_NAME} AND DEFINED ARG_DEFAULT)
        set(${VAR_NAME} "${ARG_DEFAULT}" CACHE STRING "Set to default value for ${VAR_NAME}")
    endif()

    # If required and still undefined, fail
    if(ARG_REQUIRED AND NOT DEFINED ${VAR_NAME})
        message(FATAL_ERROR "Required variable '${VAR_NAME}' is not set. Set it via -D${VAR_NAME}=... or export ${VAR_NAME}=...")
    endif()
endfunction()

get_cross_var(CROSS_SYSROOT REQUIRED)
get_cross_var(IMI_LLVM_PATH REQUIRED)
get_cross_var(CROSS_ARCH REQUIRED)
get_cross_var(CROSS_TRIPLE REQUIRED)
get_cross_var(CROSS_FLAGS DEFAULT "")
get_cross_var(CROSS_COMP REQUIRED)
get_cross_var(CROSS_CPU REQUIRED)
get_cross_var(CROSS_QEMU_PATH DEFAULT "")
get_cross_var(CROSS_TOOLCHAIN DEFAULT "")
get_cross_var(RVV_VLEN DEFAULT "")
get_cross_var(BINSTRIP DEFAULT "")
get_cross_var(DEBUG_SYMS DEFAULT "")


set(CMAKE_SYSROOT "${CROSS_SYSROOT}")
set(TARGET_TRIPLE "${CROSS_TRIPLE}")

set(BAREMETAL OFF)
if (TARGET_TRIPLE MATCHES "linux")
    set(CMAKE_SYSTEM_NAME Linux)
elseif(TARGET_TRIPLE MATCHES "elf")
    message(FATAL_ERROR "Baremetal is not currently supported")
else()
    message(FATAL_ERROR "Target triple is neither baremetal or linux, and is unsupported: ${TARGET_TRIPLE}")
endif()

set(CMAKE_SYSTEM_PROCESSOR "${CROSS_ARCH}")
if(NOT "${CROSS_ARCH}" STREQUAL "x86_64")
    set(CROSS_COMPILING TRUE)
else()
    set(CROSS_COMPILING TRUE)
endif()
# Set sysroot path for cross-compilation (typically /usr/<target-triple> for Ubuntu packages)

if (BAREMETAL)
#    set(CMAKE_SYSROOT "/usr/${TARGET_TRIPLE}/${TARGET_TRIPLE}")
    message(FATAL_ERROR "Baremetal is not currently supported")
elseif(NOT CMAKE_SYSROOT)
    set(CMAKE_SYSROOT "/")
endif()
set(CROSS_ARCH_FLAGS_TMP "")
# Architecture-specific compiler flags
if("${CROSS_ARCH}" STREQUAL "aarch64")
    # aarch64-specific settings
    # For arm, we need to get the architecture version as well
    get_cross_var(CROSS_ARCH_VARIANT REQUIRED)
    list(APPEND CROSS_ARCH_FLAGS_TMP "-mcpu=${CROSS_CPU} -march=${CROSS_ARCH_VARIANT}")
elseif("${CROSS_ARCH}" STREQUAL "riscv64")
    # riscv64-specific settings
    if ("${CROSS_CPU}" MATCHES "rv64")
        list(APPEND CROSS_ARCH_FLAGS_TMP "-march=${CROSS_CPU}")
        if(DEFINED RVV_VLEN)
            list(APPEND CROSS_ARCH_FLAGS_TMP "-mrvv-vector-bits=${RVV_VLEN}")
        endif()
    else()
        list(APPEND CROSS_ARCH_FLAGS_TMP "-mcpu=${CROSS_CPU}")
    endif()
    list(APPEND CROSS_ARCH_FLAGS_TMP "-mabi=lp64d")
else()
    # x86_64-specific settings
    list(APPEND CROSS_ARCH_FLAGS_TMP "-march=x86-64")
    list(APPEND CROSS_ARCH_FLAGS_TMP "-mtune=${CROSS_CPU}")
endif()
list(JOIN CROSS_ARCH_FLAGS_TMP " " CROSS_ARCH_FLAGS)

if("${CROSS_COMP}" STREQUAL "clang")
    if (DEFINED IMI_LLVM_PATH)
        set(CMAKE_C_COMPILER "${IMI_LLVM_PATH}/bin/clang")
        set(CMAKE_ASM_COMPILER "${IMI_LLVM_PATH}/bin/clang")
        set(CMAKE_CXX_COMPILER "${IMI_LLVM_PATH}/bin/clang++")
    else()
        set(CMAKE_C_COMPILER "/opt/tools/bin/clang-imi")
        set(CMAKE_ASM_COMPILER "/opt/tools/bin/clang-imi")
        set(CMAKE_CXX_COMPILER "/opt/tools/bin/clang++-imi")
    endif()

    set(CLANG_CROSS_FLAGS "")

    # Add cross flags to compiler and linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${CLANG_CROSS_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CLANG_CROSS_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CLANG_CROSS_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
    if(CROSS_COMPILING)
        # For clang, we use the target binutils
        if (DEFINED IMI_LLVM_PATH)
            set(CMAKE_AR "${IMI_LLVM_PATH}/bin/llvm-ar")
            set(CMAKE_RANLIB "${IMI_LLVM_PATH}/bin/llvm-ranlib")
            set(CMAKE_STRIP "${IMI_LLVM_PATH}/bin/llvm-strip")
            set(CMAKE_OBJCOPY "${IMI_LLVM_PATH}/bin/llvm-objcopy")
            set(CMAKE_OBJDUMP "${IMI_LLVM_PATH}/bin/llvm-objdump")
            set(CMAKE_NM "${IMI_LLVM_PATH}/bin/llvm-nm")
            set(CMAKE_READELF "${IMI_LLVM_PATH}/bin/llvm-readelf")
        else()
            set(CMAKE_AR llvm-ar-imi)
            set(CMAKE_RANLIB llvm-ranlib-imi)
            set(CMAKE_STRIP llvm-strip-imi)
            set(CMAKE_OBJCOPY llvm-objcopy-imi)
            set(CMAKE_OBJDUMP llvm-objdump-imi)
            set(CMAKE_NM llvm-nm-imi)
            set(CMAKE_READELF llvm-readelf-imi)
        endif()
    endif()
elseif ("${CROSS_COMP}" STREQUAL "gcc")
    # GCC compiler setup: this is all reusable for both x86 and when cross-compiling
    set(CMAKE_C_COMPILER ${CROSS_TRIPLE}-gcc)
    set(CMAKE_CXX_COMPILER ${CROSS_TRIPLE}-g++)
    set(CMAKE_ASM_COMPILER ${CROSS_TRIPLE}-gcc)
    # Set binutils tools
    set(CMAKE_AR ${CROSS_TRIPLE}-ar)
    set(CMAKE_RANLIB ${CROSS_TRIPLE}-ranlib)
    set(CMAKE_STRIP ${CROSS_TRIPLE}-strip)
    set(CMAKE_OBJCOPY ${CROSS_TRIPLE}-objcopy)
    set(CMAKE_OBJDUMP ${CROSS_TRIPLE}-objdump)
    set(CMAKE_LINKER ${CROSS_TRIPLE}-ld)
    set(CMAKE_NM ${CROSS_TRIPLE}-nm)
    set(CMAKE_READELF ${CROSS_TRIPLE}-readelf)
endif()

set(CMAKE_CXX_COMPILER_TARGET "${TARGET_TRIPLE}")
set(CMAKE_C_COMPILER_TARGET "${TARGET_TRIPLE}")
set(CMAKE_ASM_COMPILER_TARGET "${TARGET_TRIPLE}")

if (BAREMETAL)
    message(FATAL_ERROR "Baremetal is not currently supported")
endif()

# Add architecture-specific flags and user-specified flags to compiler flags
set(CMAKE_C_FLAGS "${CROSS_ARCH_FLAGS} ${CROSS_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CROSS_ARCH_FLAGS} ${CROSS_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS "${CROSS_ARCH_FLAGS} ${CROSS_FLAGS} ${CMAKE_ASM_FLAGS}")
# Configure search paths for cross-compilation
if(CROSS_COMPILING)
    set(CMAKE_FIND_ROOT_PATH "${CROSS_SYSROOT}")
    # Set find root path to sysroot
    if (BAREMETAL)
        message(FATAL_ERROR "Baremetal is not currently supported")
    else()
        set(CMAKE_PREFIX_PATH "${CMAKE_SYSROOT}")
        # set(CMAKE_INCLUDE_PATH "${CMAKE_SYSROOT}/usr/include/${CROSS_TRIPLE}")
        # set(CMAKE_LIBRARY_PATH "${CMAKE_SYSROOT}/usr/lib/${CROSS_TRIPLE}")
        # set(CMAKE_PROGRAM_PATH "${CMAKE_SYSROOT}/usr/bin/${CROSS_TRIPLE}")
    endif()
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

    # Set emulator path
    if(NOT "${CROSS_ARCH}" STREQUAL "x86_64")
        set(CMAKE_CROSSCOMPILING_EMULATOR "${CROSS_QEMU_PATH}")
    endif()

    # Set pkg-config for cross-compilation
    set(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config")
    set(ENV{PKG_CONFIG_DIR} "")
    # Set PKG_CONFIG_LIBDIR to the most common paths
    set(PKG_CONFIG_PATHS "")
    foreach(PKG_PATH
        "/usr/lib/${TARGET_TRIPLE}/pkgconfig"
        "/usr/share/pkgconfig")
        if(EXISTS "${PKG_PATH}")
            set(PKG_CONFIG_PATHS "${PKG_CONFIG_PATHS}:${PKG_PATH}")
        endif()
    endforeach()
    # Remove leading colon if present
    if(PKG_CONFIG_PATHS)
        string(SUBSTRING "${PKG_CONFIG_PATHS}" 1 -1 PKG_CONFIG_PATHS)
        set(ENV{PKG_CONFIG_LIBDIR} "${PKG_CONFIG_PATHS}")
        set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
    endif()
endif()

set(CROSS_STATIC OFF CACHE BOOL "Build static libraries")

if(CROSS_STATIC)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  list(APPEND CMAKE_CONFIGURATION_TYPES "Static")
  set_property(GLOBAL PROPERTY LINK_SEARCH_START_STATIC 1)
  set_property(GLOBAL PROPERTY LINK_SEARCH_END_STATIC 1)
  string(REPLACE "-fPIC" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fPIE" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-pie" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fPIC" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fPIE" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-pie" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
# Additional flags for Release builds
if(DEFINED BINSTRIP)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -s -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG")
elseif(DEFINED DEBUG_SYMS)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -g -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -DNDEBUG")
else()
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

# Additional flags for Debug builds
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# Print toolchain configuration status
message(STATUS "Toolchain initialized for ${CROSS_ARCH} using ${CROSS_COMP}")
message(STATUS "Target Triple: ${TARGET_TRIPLE}")
if(CROSS_COMPILING)
    message(STATUS "Cross-compiling: YES")
    message(STATUS "Sysroot: ${CMAKE_SYSROOT}")
else()
    message(STATUS "Cross-compiling: NO (native build)")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CPU: ${CROSS_CPU}")
message(STATUS "Architecture flags: ${CROSS_ARCH_FLAGS}")
message(STATUS "Additional flags: ${CROSS_FLAGS}")
