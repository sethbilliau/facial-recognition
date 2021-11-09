FROM ubuntu:latest
# Fix timezone issue
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install git wget vim gcc g++ imagemagick -y
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

# RUN apt-get update && apt-get install -y \
#         python-dev python-pip python-setuptools \
#         libffi-dev libxml2-dev libxslt1-dev \
#         libtiff4-dev libjpeg8-dev zlib1g-dev libfreetype6-dev \
#         liblcms2-dev libwebp-dev tcl8.5-dev tk8.5-dev python-tk

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
WORKDIR /app
COPY . .
RUN python3 -m pip install -r requirements.txt
# RUN python -m pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python -m pip install facenet-pytorch==2.4.1
RUN python3 -m pip install face_recognition==1.2.2
# RUN pip install Pillow

# Now install Jupyter and spin up a server
RUN pip  install jupyter jupyterlab --verbose
RUN jupyter lab --generate-config
RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

# # Update apt-get and install some stuff
# RUN apt-get -y update
# RUN apt-get install git wget vim gcc g++ -y
# RUN apt-get install -y \
#     imagemagick libmagickwand-dev --no-install-recommends \
#     && pecl install imagick \
#     && docker-php-ext-enable imagick
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install -y --fix-missing \
#     build-essential \
#     cmake \
#     gfortran \
#     git \
#     wget \
#     curl \
#     graphicsmagick \
#     libgraphicsmagick1-dev \
#     libatlas-base-dev \
#     libavcodec-dev \
#     libavformat-dev \
#     libgtk2.0-dev \
#     libjpeg-dev \
#     liblapack-dev \
#     libswscale-dev \
#     pkg-config \
#     python3-dev \
#     python3-numpy \
#     software-properties-common \
#     zip \
#     && apt-get clean && rm -rf /tmp/* /var/tmp/*

# RUN cd ~ && \
#     mkdir -p dlib && \
#     git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
#     cd  dlib/ && \
#     python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# # Now copy over the python requirements and install them
# WORKDIR /app
# COPY . .
# RUN python -m pip install --upgrade pip
# RUN python -m pip install -r requirements.txt
# # RUN python -m pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# # RUN python -m pip install facenet-pytorch==2.4.1
# RUN python -m pip install face_recognition==1.2.2

# # Now install Jupyter and spin up a server
# RUN pip3 install jupyter jupyterlab --verbose
# RUN jupyter lab --generate-config
# RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"
# CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]