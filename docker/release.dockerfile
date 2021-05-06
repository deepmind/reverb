# Run the following commands in order:
#
# REVERB_DIR="/tmp/reverb"  # (change to the cloned reverb directory, e.g. "$HOME/reverb")
# docker build --tag tensorflow:reverb_release - < "$REVERB_DIR/docker/release.dockerfile"
# docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb \
#   -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro \
#   --name reverb_release tensorflow:reverb_release bash
#
# Test that everything worked:
#
# bazel test -c opt --copt=-mavx --config=manylinux2010 --test_output=errors //reverb/...

ARG cpu_base_image="tensorflow/tensorflow:2.2.0-custom-op-ubuntu16"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Reverb Team <no-reply@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="tensorflow/tensorflow:2.1.0-custom-op-ubuntu16"
ARG base_image=$cpu_base_image
ARG tensorflow_pip="tf-nightly"
ARG python_version="python3.6"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        aria2 \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python3-dev \
        python3.6-dev \
        python3.7-dev \
        python3.8-dev \
        # Needed due to python3.8 apt packaging issue.
        python3.8-distutils \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py

ARG pip_dependencies=' \
      absl-py \
      contextlib2 \
      dataclasses \
      dm-tree>=0.1.5 \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      platform \
      portpicker'

# TODO(b/154930404): Update to 2.2.0 once it's out.  May need to
# cut a branch to make changes that allow us to build against 2.2.0 instead
# of tf-nightly due to API changes.
RUN for python in ${python_version}; do \
    $python get-pip.py && \
    $python -mpip uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu && \
    $python -mpip --no-cache-dir install ${tensorflow_pip} --upgrade && \
    $python -mpip --no-cache-dir install $pip_dependencies; \
  done
RUN rm get-pip.py

# Needed until this is included in the base TF image.
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.8" "/dt7/usr/include/x86_64-linux-gnu/python3.8"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.8" "/dt8/usr/include/x86_64-linux-gnu/ppython3.8"


WORKDIR "/tmp/reverb"

CMD ["/bin/bash"]
