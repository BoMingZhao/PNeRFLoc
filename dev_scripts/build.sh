
cd submodule/opencv-python/
PYTHON_EXEC=$1
cuda_arch=$2
# rm -rf build
# mkdir build
# cd build

# cmake -D CMAKE_BUILD_TYPE=RELEASE \
#       -D CMAKE_INSTALL_PREFIX=/usr/local \
#       -D INSTALL_PYTHON_EXAMPLES=OFF \
#       -D INSTALL_C_EXAMPLES=OFF \
#       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#       -D PYTHON_EXECUTABLE=/mnt/nas_9/group/zhaoboming/conda/envs/PNeRFLoc_4090/bin/python3 \
#       -D BUILD_EXAMPLES=ON \
#       -D WITH_CUDA=ON \
#       -D CUDA_ARCH_BIN=8.9 \
#       -D CUDA_FAST_MATH=ON \
#       -D WITH_CUBLAS=ON \
#       -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
#       -D OpenCL_LIBRARY=/usr/local/cuda-11.8/lib64/libOpenCL.so \
#       -D OpenCL_INCLUDE_DIR=/usr/local/cuda-11.8/include/ \
      # ..
pip install scikit-build
ENABLE_CONTRIB=1 python setup.py bdist_wheel -- \
    -DWITH_CUDA=ON \
    -DENABLE_FAST_MATH=1 \
    -DPYTHON_EXECUTABLE=$PYTHON_EXEC \
    -DCUDA_FAST_MATH=1 \
    -DWITH_CUBLAS=1 \
    -DCUDA_ARCH_BIN=$cuda_arch -- \
    -j8

# export ENABLE_CONTRIB=1
# export CMAKE_ARGS="-DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUDA=ON -DCUDA_ARCH_BIN=8.9 -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_CUBLAS=ON "

# pip install --no-binary opencv-python opencv-python
# pip wheel . --verbose
# make -j8
# sudo ldconfig
# sudo make install