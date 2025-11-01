#!/bin/bash
set -e

echo "Installing build tools first..."
pip install --upgrade pip setuptools wheel

echo "Installing core dependencies..."
pip install Django Pillow

echo "Installing scientific packages..."
pip install numpy

echo "Installing TensorFlow..."
pip install tensorflow==2.13.0 tensorflow-macos==2.13.0

echo "Build completed successfully!"