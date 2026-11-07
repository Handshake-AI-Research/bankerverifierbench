# Build stage: compile the setuid MCP wrapper.
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY mcp-server/wrapper.c /tmp/wrapper.c
RUN gcc -o /tmp/mcp-server-run /tmp/wrapper.c

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MCP_STDIO_SERVERS=[{\"name\":\"mcp-server\",\"command\":\"/usr/bin/mcp-server\"}]

RUN apt-get update && \
    apt-get install -y bash curl git nodejs npm sudo bzip2 zstd libxcb1 libgomp1 libreoffice tmux unzip poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN printf '%s\n' \
      'Defaults !requiretty' \
      'Defaults secure_path="/opt/agent-python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
      'Defaults env_keep += "BVB_TOOL_ARCHIVE_INDEX BVB_TOOL_ARCHIVE_ROOT BVB_TOOL_CACHE_ROOT BVB_TOOL_CACHE_MAX_BYTES"' \
      > /etc/sudoers.d/00-bvb-runtime && \
    chmod 440 /etc/sudoers.d/00-bvb-runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN useradd -m -u 10000 -d /home/environment -s /bin/bash environment && \
    useradd -m -u 10001 -d /home/agent -s /bin/bash agent && \
    useradd -m -u 10002 -d /home/verifier -s /bin/bash verifier && \
    useradd -m -u 10003 -d /home/sandbox -s /bin/bash sandbox && \
    usermod -aG agent environment && \
    usermod -aG agent verifier && \
    usermod -aG verifier sandbox

RUN echo "verifier ALL=(sandbox) NOPASSWD: ALL" >> /etc/sudoers.d/verifier && \
    chmod 440 /etc/sudoers.d/verifier

RUN mkdir -p /home/agent/workspace && chown -R agent:agent /home/agent && \
    chmod 2775 /home/agent/workspace && \
    echo "umask 002" >> /home/agent/.bashrc && \
    echo "umask 002" >> /home/agent/.profile
RUN mkdir -p /home/verifier/workspace && chown -R verifier:verifier /home/verifier && \
    usermod -aG verifier environment && \
    chmod 2775 /home/verifier/workspace && \
    echo "umask 002" >> /home/verifier/.bashrc && \
    echo "umask 002" >> /home/verifier/.profile

RUN mkdir /installed-agent && chown -R agent:agent /installed-agent && chmod -R a+rw /installed-agent
RUN mkdir /oracle && chown -R agent:agent /oracle && chmod -R a+rw /oracle

COPY --chown=environment:environment mcp-server /opt/mcp-server
USER environment
RUN cd /opt/mcp-server && uv sync
USER root
RUN chmod 700 /opt/mcp-server

COPY --from=build /tmp/mcp-server-run /usr/bin/mcp-server
RUN chown environment:environment /usr/bin/mcp-server && \
    chmod u+s /usr/bin/mcp-server

# RewardKit and its agent judge are image dependencies, not runtime downloads.
# Keep both versions explicit because they are part of the verifier semantics.
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv tool install 'harbor-rewardkit==0.1.2' && \
    chmod -R a+rX /opt/uv-tools && \
    ([ -d /opt/uv-python ] && chmod -R a+rX /opt/uv-python || true)
RUN npm install -g @anthropic-ai/claude-code@2.1.246 opencode-ai@latest

COPY agent-python-requirements.txt /tmp/agent-python-requirements.txt
RUN UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv venv /opt/agent-python --python 3.14 && \
    VIRTUAL_ENV=/opt/agent-python uv pip install -r /tmp/agent-python-requirements.txt && \
    chmod -R a+rX /opt/agent-python
ENV VIRTUAL_ENV=/opt/agent-python
ENV PATH="/opt/agent-python/bin:$PATH"
RUN /opt/agent-python/bin/python -c \
      "import openpyxl, pandas, pptx, pdfplumber, pypdf, reportlab" && \
    sudo -n -u verifier env HOME=/home/verifier claude --version && \
    sudo -n -u verifier env HOME=/home/verifier rewardkit --help >/dev/null

COPY --chown=agent:agent input/ /home/agent/workspace/
RUN chmod 2775 /home/agent/workspace

WORKDIR /home/agent/workspace
RUN mkdir -p /var/cache/bvb-tools && \
    chown environment:environment /var/cache/bvb-tools && \
    chmod 700 /var/cache/bvb-tools
USER agent
CMD ["/bin/bash"]
