# Run the following commands in order:
#
# REVERB_DIR="/tmp/reverb"  # (change to the cloned reverb directory, e.g. "$HOME/reverb")
# docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/dev.dockerfile"
# docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb \
#   -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro \
#   --name reverb tensorflow:reverb bash
#
# Test that everything worked:
#
# bazel test -c opt --test_output=streamed //reverb:tf_client_test

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Reverb Team <no-reply@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
ARG tensorflow_pip="tf-nightly"
ARG python_version="python3.6"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        aria2 \
        build-essential \
        curl \
        gdb \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python3-distutils \
        python3.6-dev \
        python3.7-dev \
        python3.8-dev \
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

ARG bazel_version=2.2.0
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Add support for Bazel autocomplete, see
# https://docs.bazel.build/versions/master/completion.html for instructions.
RUN cp /usr/local/lib/bazel/bin/bazel-complete.bash /etc/bash_completion.d
# TODO(b/154932078): This line should go into bashrc.
# NOTE(ebrevdo): RUN source doesn't work.  Disabling the command below for now.
# RUN source /etc/bash_autcompletion.d/bazel-complete.bash

ARG pip_dependencies=' \
      contextlib2 \
      dm-tree>=0.1.5 \
      dataclasses \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      platform \
      portpicker'


# So dependencies are installed for the supported Python versions
RUN for python in ${python_version}; do \
    $python get-pip.py && \
    $python -mpip uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu && \
    $python -mpip --no-cache-dir install ${tensorflow_pip} --upgrade && \
    $python -mpip --no-cache-dir install $pip_dependencies; \
  done

RUN rm get-pip.py

# bazel assumes the python executable is "python".
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR "/tmp/reverb"

CMD ["/bin/bash"]
