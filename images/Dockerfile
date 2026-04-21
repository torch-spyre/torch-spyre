FROM us.icr.io/wxpe-cicd-internal/amd64/torch-aiu-runtime-dev:latest
USER root
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN cd "$HOME" && \
    source "$HOME/.bashrc" && \
    git clone https://github.com/torch-spyre/torch-spyre && \
    cd torch-spyre/ && \
    uv sync --all-extras --active --inexact --reinstall-package torch-spyre -v
USER senuser
