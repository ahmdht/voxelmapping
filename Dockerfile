# Dockerfile for VoxelMapping with Mech-Eye SDK
# Supports: voxelmapping library, interactive_capture_mecheye.py, camera tests
#
# Build:
#   docker build -t voxelmapping:latest .
#
# Run interactively:
#   docker run -it --rm --privileged --network=host \
#       -v /dev:/dev \
#       -v $(pwd)/captures:/app/captures \
#       voxelmapping:latest
#
# Run with X11 display (for Open3D visualization):
#   docker run -it --rm --privileged --network=host \
#       -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -v $(pwd)/captures:/app/captures \
#       voxelmapping:latest

FROM ubuntu:22.04

LABEL maintainer="ahmad.hoteit"
LABEL description="VoxelMapping with Mech-Eye SDK for 3D point cloud processing"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ============================================================================
# BASE SYSTEM PACKAGES
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unzip \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# UPGRADE G++ TO VERSION 13 (Required for Mech-Eye SDK)
# ============================================================================
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y g++-13 gcc-13 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    rm -rf /var/lib/apt/lists/*

# Verify g++ version
RUN g++ --version

# ============================================================================
# BUILD TOOLS AND DEPENDENCIES
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    # PCL dependencies
    libpcl-dev \
    libpcl-common1.12 \
    libpcl-io1.12 \
    libpcl-filters1.12 \
    # Eigen3
    libeigen3-dev \
    # pybind11
    pybind11-dev \
    python3-pybind11 \
    # MPI (required by CMakeLists.txt)
    libopenmpi-dev \
    openmpi-bin \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    # OpenCV (for Mech-Eye samples)
    libopencv-dev \
    python3-opencv \
    # USB/Network for camera access
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    # Visualization dependencies (for Open3D)
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxrender1 \
    libxext6 \
    libsm6 \
    # Network utilities (required by Mech-Eye SDK)
    iputils-ping \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PYTHON PACKAGES
# ============================================================================
# Pin numpy<2 to avoid ABI mismatch with compiled extensions (cv2, etc.)
RUN pip3 install --no-cache-dir \
    'numpy<2' \
    scipy \
    pytest \
    open3d \
    pyyaml

# ============================================================================
# MECH-EYE SDK INSTALLATION
# ============================================================================
# Option 1: Download SDK manually and place in build context as mecheye_sdk.deb
# Option 2: Mount SDK from host at runtime
# Option 3: Provide download URL via build arg (for private builds)
#
# Download SDK from: https://downloads.mech-mind.com/
# Or use: wget -O mecheye_sdk.deb "<YOUR_SDK_URL>"

WORKDIR /tmp

# Install from local .deb if provided in build context
COPY mecheye_sdk.deb* /tmp/
RUN if [ -f /tmp/mecheye_sdk.deb ]; then \
        dpkg -i /tmp/mecheye_sdk.deb || apt-get install -f -y && \
        rm -f /tmp/mecheye_sdk.deb && \
        ldconfig; \
    else \
        echo "No mecheye_sdk.deb found. SDK will need to be mounted at runtime."; \
    fi

# Install Python Mech-Eye API
RUN pip3 install --no-cache-dir MechEyeApi || echo "MechEyeApi pip install skipped (install SDK first)"

# Verify Mech-Eye Python installation
RUN python3 -c "from mecheye.shared import *; print('Mech-Eye Python API installed successfully')" || echo "Warning: Mech-Eye Python import test skipped - install SDK and rebuild"

# ============================================================================
# WORKSPACE SETUP (source files mounted at runtime)
# ============================================================================
WORKDIR /app

# Create directory structure (files will be mounted)
RUN mkdir -p /app/build /app/captures

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
ENV PYTHONPATH="/app/build:/app/python:/diana_api"
ENV LD_LIBRARY_PATH="/diana_lib:/opt/mech-mind/mech-eye-sdk/lib:/usr/local/lib"

WORKDIR /app

# ============================================================================
# ENTRYPOINT SCRIPT
# ============================================================================
RUN echo '#!/bin/bash\n\
echo "=== VoxelMapping Container ==="\n\
echo ""\n\
echo "Build voxelmapping (if needed):"\n\
echo "  cd /app/build && cmake .. && make -j$(nproc)"\n\
echo ""\n\
echo "Run interactive capture:"\n\
echo "  python3 interactive_capture_mecheye.py --out captures --voxel-size 0.005"\n\
echo ""\n\
echo "Run tests:"\n\
echo "  cd /app/build && ctest"\n\
echo ""\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
