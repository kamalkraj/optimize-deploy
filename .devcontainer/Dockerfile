ARG VERSION="22.07-py3"
FROM nvcr.io/nvidia/tensorrt:${VERSION}

# https://code.visualstudio.com/docs/remote/containers-advanced#_creating-a-nonroot-user
ARG USERNAME=user-vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo bash locales \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && locale-gen en_US.UTF-8

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV NO_COMPILER_INSTALLATION=1

RUN pip3 install -U pip && pip install nebullvm==0.4.0
RUN python -c "import nebullvm"
RUN pip3 uninstall onnxruntime && pip3 install onnxruntime-gpu

CMD ["/bin/bash"]
