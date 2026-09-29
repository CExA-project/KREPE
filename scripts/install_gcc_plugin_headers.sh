#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

set -e

USE_EXTERNAL_GMP=0
USE_EXTERNAL_MPFR=0
USE_EXTERNAL_MPC=0
DESTINATION=$(pwd)/plugin_headers
CONFIGURE_ARGS=
CXX="${CXX:-g++}"

function help {
  cat <<EOF
usage: $0 [--gmp-root <path>] [--mpfr-root <path>] [--mpc-root <path>] [--destination <path>]

options:
  --gmp-root Path to an existing installation of GMP. If specified, GMP won't be built alongside the headers, otherwise GMP will be built and installed with the headers
  --mpfr-root Path to an existing installation of MPFR. If specified, MPFR won't be built alongside the headers
  --mpc-root Path to an existing installation of MPC. If specified, MPC won't be built alongside the headers
  --destination Specifies the path where the headers will be installed
  -h, --help Print this help message

notes:
  The compiler for which the headers are built can be specified using the CXX environment variable, otherwise g++ will be used.
EOF
  exit 0
}

while [ $# -gt 0 ];
do
  case $1 in
  --gmp-root)
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-gmp=$2"
    USE_EXTERNAL_GMP=1
    shift 2
    ;;
  --gmp-root=*)
    ARG=$(echo "$1" | sed 's/[^=]\+=\(.*\)/\1/')
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-gmp=$ARG"
    USE_EXTERNAL_GMP=1
    shift
    ;;
  --mpfr-root)
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-mpfr=$2"
    USE_EXTERNAL_MPFR=1
    shift 2
    ;;
  --mpfr-root=*)
    ARG=$(echo "$1" | sed 's/[^=]\+=\(.*\)/\1/')
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-mpfr=$ARG"
    USE_EXTERNAL_MPFR=1
    shift
    ;;
  --mpc-root)
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-mpc=$2"
    USE_EXTERNAL_MPC=1
    shift 2
    ;;
  --mpc-root=*)
    ARG=$(echo "$1" | sed 's/[^=]\+=\(.*\)/\1/')
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-mpc=$ARG"
    USE_EXTERNAL_MPC=1
    shift
    ;;
  --destination)
    DESTINATION="$2"
    shift 2
    ;;
  --destination=*)
    ARG=$(echo "$1" | sed 's/[^=]\+=\(.*\)/\1/')
    DESTINATION="$ARG"
    shift
    ;;
  -h|--help)
    help
    ;;
  *)
    echo "unknown flag '$1'"
    exit 1
    ;;
  esac
done

DESTINATION=$(realpath "$DESTINATION")
echo "Installation directory: $DESTINATION"

GCC_VERSION=$(echo "$($CXX -dumpfullversion)" | sed -e 's/[0-9]\+$/0/')
GCC_TARGET=$($CXX -dumpmachine)

CONFIGURE_ARGS="$CONFIGURE_ARGS --enable-plugin --enable-languages=c,c++ --disable-bootstrap --disable-multilib --disable-nls --disable-lto --without-isl --target=${GCC_TARGET} --host=${GCC_TARGET} --build=${GCC_TARGET}"

WORKDIR=$(mktemp -d)

function cleanup {
  echo "Deleting working directory $WORKDIR"
  rm -rf $WORKDIR
}

trap cleanup EXIT

cd $WORKDIR
echo "Fetching gcc-${GCC_VERSION}.tar.xz..."
wget https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz
echo "Extracting gcc-${GCC_VERSION}.tar.xz..."
tar xf gcc-${GCC_VERSION}.tar.xz

cd gcc-${GCC_VERSION}
echo "Dowloading prerequisites"
./contrib/download_prerequisites
rm -rf isl*
if [ $USE_EXTERNAL_MPFR -eq 1 ]; then
  rm -rf mpfr*
fi
if [ $USE_EXTERNAL_MPC -eq 1 ]; then
  rm -rf mpc*
fi
if [ $USE_EXTERNAL_GMP -eq 1 ]; then
  rm -rf gmp*
else
  pushd gmp
  echo "Installing GMP"
  ./configure prefix="$DESTINATION"
  make
  # make check
  make install
  popd
  rm -rf gmp*
  CONFIGURE_ARGS="$CONFIGURE_ARGS --with-gmp=$DESTINATION"
fi

mkdir build
cd build
echo "Installing the headers"
../configure $CONFIGURE_ARGS prefix="$DESTINATION"
make MAKEINFO=true configure-gcc
make MAKEINFO=true all-libiberty all-libcpp all-libdecnumber all-libbacktrace
cd gcc
make install-plugin build_libsubdir= DESTDIR="$DESTINATION"
PLUGIN_INC_DIR=$(dirname "$(find "$DESTINATION" -path "*/plugin/include/gcc-plugin.h" -type f)")
ln -s "$PLUGIN_INC_DIR" "$DESTINATION/plugin"

echo "Installation successful"
