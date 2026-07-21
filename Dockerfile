# Optional dev environment for contributors who don't want to install Hugo natively.
# Installs the same Hugo Extended version pinned in .github/workflows/deploy.yml.
FROM debian:bookworm-slim

ARG HUGO_VERSION=0.160.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates=20230311+deb12u1 \
    wget=1.21.3-1+deb12u1 \
    git=1:2.39.5-0+deb12u3 \
    && wget -O /tmp/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
    && dpkg -i /tmp/hugo.deb \
    && rm /tmp/hugo.deb \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /site
EXPOSE 1313

ENTRYPOINT ["hugo"]
CMD ["server", "--bind", "0.0.0.0", "--disableFastRender", "--buildDrafts"]
