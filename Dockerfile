# Minimal source-build bootstrap for the public GRAIL runtime.
# For the prebuilt three-environment images, see docs/source/installation.md.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ gcc git git-lfs wget curl unzip \
    ffmpeg xvfb xauth \
    libeigen3-dev \
    libboost-all-dev \
    gosu \
    xvfb \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge.  Unlike the current Miniconda bootstrap, this remains
# non-interactive in container builds and does not require accepting Anaconda
# channel terms before `conda create`.
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

# Create grail conda environment with Python 3.10
RUN conda create -y -n grail python=3.10

# Install the CUDA stack used by the released GRAIL runtime.  Pinning avoids
# silently moving to a newer PyTorch ABI when the image is rebuilt.
RUN conda run -n grail pip install \
    'torch==2.7.1' 'torchvision==0.22.1' \
    --index-url https://download.pytorch.org/whl/cu128

# Native SOMA is the default human representation.  uv provisions the pinned
# Python 3.12 SOMA Retargeter environment after the repo is bind-mounted; its
# source-bound target generator uses the original GMR G1+DEx3 MJCF and meshes.
# Keeping that lockfile environment separate avoids contaminating GRAIL's
# Python 3.10 dependency set.
RUN conda run -n grail python -m pip install \
    'py-soma-x==0.3.3' 'warp-lang==1.8.0' 'uv>=0.9,<1'
RUN conda run -n grail python -c \
    "import importlib.metadata as m, soma, torch; \
assert torch.__version__ == '2.7.1+cu128'; \
assert soma.__version__ == '0.3.3'; \
assert m.version('warp-lang') == '1.8.0'; \
assert tuple(map(int, m.version('uv').split('.')[:1])) < (1,)"

# Headless rendering environment
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless

# Activate grail env by default in bash
RUN echo "conda activate grail" >> /root/.bashrc

# Embed entrypoint
COPY tools/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace/grail

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
