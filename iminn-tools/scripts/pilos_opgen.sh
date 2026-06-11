#!/bin/sh
set -e

cd Src/Support/uop_generation
python3 riscv_parser.py v-rv_imi_matrix
cd ../../..