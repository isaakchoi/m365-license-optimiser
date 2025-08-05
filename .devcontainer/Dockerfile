########################
##        NOTE        ##
########################
# Repeately building in a short period of time will cause servers to block requests and the build will fail.


## USE DEBIAN AS BASE IMAGE ...

FROM mcr.microsoft.com/devcontainers/base:bookworm


## GENERAL CONFIG ...

# Set to run using bash with --login mode
SHELL ["/bin/bash", "-l", "-c"]

# The username of the user that the container will open with
ARG USER=vscode

# Path to use for temporary file placement during build processes - Assume all content will be deleted after process completion
ARG TMP_DIR=/app/tmp

# A possible location to put installed binaries
ARG BINARY_INSTALL_PATH=/usr/local/bin

# Read in args given by devcontainer.json
ARG HOST_WORKSPACE_PATH
ARG REMOTE_WORKSPACE_PATH


## CATCH CONFIG ISSUES EARLY ...

# Ensure HOST_WORKSPACE_PATH was given
RUN if [ -z "${HOST_WORKSPACE_PATH}" ]; then exit 1; fi

# Ensure REMOTE_WORKSPACE_PATH was given
RUN if [ -z "${REMOTE_WORKSPACE_PATH}" ]; then exit 1; fi

# Ensure BINARY_INSTALL_PATH is on PATH
RUN \
    if [[ ":$PATH:" != *":$BINARY_INSTALL_PATH:"* ]]; then exit 1 ; fi


## INSTALL PYTHON AND PIP ...

ARG PYTHON_VERSION=3.12.10

USER ${USER}

# Install pyenv to allow installing and managing specific python versions. See: https://github.com/pyenv/pyenv
RUN \
    curl -fsSL https://pyenv.run | bash \
    && echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> "$HOME/.bashrc" \
    && echo 'eval "$(pyenv init -)"' >> "$HOME/.bashrc" \
    && echo 'eval "$(pyenv virtualenv-init -)"' >> "$HOME/.bashrc"

# Ensure python build dependencies are installed prior to installing. See: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
RUN \
    sudo apt-get update -y \
    && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        curl \
        git \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        llvm

# Intall specified python version and pip
RUN \
    # Load pyenv config into current terminal instance and install python
    export PATH="$HOME/.pyenv/bin:$PATH" \
    && eval "$(pyenv init -)" \
    && eval "$(pyenv virtualenv-init -)" \
    && pyenv install ${PYTHON_VERSION} \
    # Default `python` command to the new installation
    && pyenv global ${PYTHON_VERSION} \
    # Ensure pip is installed
    && python -m ensurepip --upgrade \
    # Ensure pip is up to date
    && python -m pip install --upgrade pip --no-input


## INSTALL PIP PACKAGES ...

ARG PRECOMMIT_VERSION=3.8.0

USER ${USER}

RUN \
    # Load pyenv config into current terminal instance and install
    export PATH="$HOME/.pyenv/bin:$PATH" \
    && eval "$(pyenv init -)" \
    && pyenv global ${PYTHON_VERSION} \
    # Install with pip
    && python -m pip install \
        pre-commit==${PRECOMMIT_VERSION}


## INSTALL GITHUB CLI ...

USER ${USER}

# Add to keychain and install
RUN \
    # As per https://github.com/cli/cli/blob/trunk/docs/install_linux.md
    (type -p wget >/dev/null || (sudo apt-get update && sudo apt-get install wget -y)) \
    && sudo mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt-get update -y \
    && sudo apt-get install gh -y

# Fix issue where different owner/contributor to bit is blocking commands...
RUN \
    git config --global --add safe.directory "${REMOTE_WORKSPACE_PATH}"


# ## INSTALL GCLOUD CLI

# USER root

# # Install gcloud cli
# RUN \
#     echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
#         | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
#         && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
#             | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
#         && apt-get update -y && apt-get install google-cloud-cli -y


# ## INSTALL NODE (24.x), NPM, YARN

# USER root

# # Install Node.js via official NodeSource setup script. This also includes npm.
# RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
#     && apt-get install -y nodejs

# # Disable Corepack download prompt
# ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0

# # Enable and prepare Corepack
# RUN corepack enable \
#     && corepack prepare yarn@1.22.22 --activate


## INSTALL POTENTIAL BUILD DEPENDENCIES (NEED TO CHECK WHAT IS ACTUALLY NEEDED) ...

USER ${USER}

RUN \
    sudo apt-get update -y \
    && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
        build-essential \
        libxi-dev \
        libglu1-mesa-dev \
        libglew-dev \
        pkg-config \
        xdg-utils
