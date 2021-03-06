# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition on the GPU
# In order to run Docker in the GPU you will need to install Nvidia-Docker: https://github.com/NVIDIA/nvidia-docker

FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

RUN apt-get update && apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install git wget vim gcc g++ imagemagick -y
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

# get dlib to work
RUN apt-get install -y --fix-missing \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    imagemagick \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Now copy over the python requirements and install them
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install face_recognition

# Copy over my stuff
WORKDIR /app
COPY . .

# Now install Jupyter and spin up a server
# RUN pip  install jupyter jupyterlab --verbose
# RUN jupyter lab --generate-config
RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

