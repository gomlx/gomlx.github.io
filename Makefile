# GoMLX documentation site — Makefile
# Usage: make <target>

HUGO        := hugo
.PHONY: help dev build sync clean

help:
	@echo ""
	@echo "  GoMLX docs site commands:"
	@echo ""
	@echo "  make dev        — start local Hugo dev server (hot reload)"
	@echo "  make build      — build production site to ./public/"
	@echo "  make sync       — pull latest docs from gomlx/gomlx (latest release by default)"
	@echo "                    Options:"
	@echo "                      make sync VERSION=v0.27.3  (specific version tag)"
	@echo "                      make sync BRANCH=main      (specific branch)"
	@echo "                      make sync COMMIT=abc1234   (specific commit hash)"
	@echo "                      make sync LOCAL_PATH=../   (local repository path)"
	@echo "  make clean      — remove ./public/ build output"
	@echo ""

PORT_OPT = 
ifdef PORT
  PORT_OPT = --port $(PORT)
endif

dev:
	$(HUGO) server $(PORT_OPT) --disableFastRender --buildDrafts

build:
	$(HUGO)

SYNC_OPTS =
ifdef VERSION
	SYNC_OPTS = -version $(VERSION)
else ifdef BRANCH
	SYNC_OPTS = -branch $(BRANCH)
else ifdef COMMIT
	SYNC_OPTS = -commit $(COMMIT)
else ifdef LOCAL_PATH
	SYNC_OPTS = -path $(LOCAL_PATH)
endif

sync:
	go run cmd/sync_docs/main.go $(SYNC_OPTS)

clean:
	rm -rf public/

# Full workflow: build
all: build
