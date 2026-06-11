#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 [options]"
    echo "Commands:"
    echo "Options:"
    echo "--rebuild       Recompile the codebase instead of building from scratch."
    echo "--install-dir   Specify the name of the install directory. Defaults to 'install-local'"
    echo "--debug         Enable a debug build for troubleshooting."
    echo "--system        Build system qemu emulator."
}

abort() {
  echo "Error: $1" >&2
  usage
  exit 1
}

QEMU_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REBUILD=false
BUILD_DIR=build
INSTALL_DIR=install-local
TARGET="linux-user"
BUILD_TYPE="user"
BUILD_ARGS=

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rebuild)
            REBUILD=true
            shift 1
        ;;
        -i|--install-dir)
            INSTALL_DIR=$2
            shift 2
        ;;
        -d|--debug)
            BUILD_ARGS=--enable-debug
            shift 1
        ;;
        -s|--system)
            INSTALL_DIR=install-sys-local
            TARGET="softmmu"
            BUILD_TYPE="sys"
            shift 1
        ;;
        -*)
            abort "Unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done
# MAKE_ARGS="-j $(nproc)"
MAKE_ARGS=""
BUILD_LIB_DIR="${BUILD_DIR}-${BUILD_TYPE}-lib"
BUILD_BIN_DIR="${BUILD_DIR}-${BUILD_TYPE}-bin"

# TODO: add checks for directory existing
compile_bin() {
    echo "Compiling from $BUILD_BIN_DIR, installing to $INSTALL_DIR"
    cd $BUILD_BIN_DIR
    make $MAKE_ARGS || abort "Error"
    make install || abort "Error"
    cd $QEMU_ROOT
    # cp -R $BUILD_USER_DIR/$INSTALL_DIR/*  $INSTALL_DIR/
    cp $BUILD_BIN_DIR/contrib/plugins/*.so  $INSTALL_DIR/plugins
}


compile_lib() {
    echo "Compiling from $BUILD_LIB_DIR, installing to $INSTALL_DIR"
    cd $BUILD_LIB_DIR
    make $MAKE_ARGS || abort "Error"
    make install || abort "Error"
    cd $QEMU_ROOT
    # cp -R $BUILD_USER_DIR/$INSTALL_DIR/*  $INSTALL_DIR/
    cp $BUILD_LIB_DIR/contrib/plugins/*.so  $INSTALL_DIR/plugins
}

add_sys_files() {
    if [[ "$BUILD_TYPE" = "sys" ]]; then
        cp $INSTALL_DIR/bin/qemu-img $INSTALL_DIR/bin/qemu-img-good
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/Makefile  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/psim*  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/fsim*  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/*.sh  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/hw/vsi/lib*.so  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/rtl_cosim*  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/Makefile.snps.QEMU  $INSTALL_DIR/rtl-qemu-shim
        cp $QEMU_ROOT/contrib/rtl-qemu-shim/env-QEMU.csh        $INSTALL_DIR/rtl-qemu-shim
    fi
}

compile() {
    compile_bin
    compile_lib
    add_sys_files
}

clean() {
    rm -rf $BUILD_BIN_DIR
    rm -rf $BUILD_LIB_DIR
    rm -rf $INSTALL_DIR
}

build() {
    clean
    mkdir -p $BUILD_BIN_DIR
    mkdir -p $BUILD_LIB_DIR
    mkdir -p $INSTALL_DIR/bin
    mkdir -p $INSTALL_DIR/share
    mkdir -p $INSTALL_DIR/include
    mkdir -p $INSTALL_DIR/plugins
    mkdir -p $INSTALL_DIR/rtl-qemu-shim

    cd $BUILD_BIN_DIR
    $QEMU_ROOT/configure --prefix="${QEMU_ROOT}/${INSTALL_DIR}" --target-list="riscv64-${TARGET}" --extra-ldflags="-Wl,--no-as-needed,-ldl" $BUILD_ARGS

    cd $QEMU_ROOT/$BUILD_LIB_DIR
    $QEMU_ROOT/configure --prefix="${QEMU_ROOT}/${INSTALL_DIR}" --target-list="riscv64-lib-${TARGET}" --extra-ldflags="-Wl,--no-as-needed,-ldl" $BUILD_ARGS

    cd $QEMU_ROOT
    compile
}

if [[ "$REBUILD" = true ]]; then
    compile
else
    build
fi