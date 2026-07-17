#!/usr/bin/env bash
set -euo pipefail

# GraphRAG V2 — Full Knowledge Base Build Pipeline
# Usage: bash scripts/build_pipeline.sh [options]
#
# Options:
#   --source DIR          Document source directory (default: data/raw)
#   --config FILE         Config file path (default: configs/default.yaml)
#   --skip-ingest         Skip document ingestion step
#   --skip-graph          Skip knowledge graph build step
#   --skip-communities    Skip community detection & summarization step
#   -h, --help            Show this help message

SOURCE="data/raw"
CONFIG="configs/default.yaml"
SKIP_INGEST=false
SKIP_GRAPH=false
SKIP_COMMUNITIES=false

usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | grep -v '^#!' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)        SOURCE="$2"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --skip-ingest)   SKIP_INGEST=true; shift ;;
        --skip-graph)    SKIP_GRAPH=true; shift ;;
        --skip-communities) SKIP_COMMUNITIES=true; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

PIPELINE_START=$(date +%s)

echo "========================================"
echo "  GraphRAG V2 — Knowledge Base Pipeline"
echo "========================================"
echo "  Source  : $SOURCE"
echo "  Config  : $CONFIG"
echo "  Skip    : ingest=$SKIP_INGEST  graph=$SKIP_GRAPH  communities=$SKIP_COMMUNITIES"
echo "========================================"
echo ""

# Step 1: Ingest
if [ "$SKIP_INGEST" = false ]; then
    echo ">>> Step 1/3: Ingesting documents..."
    T0=$(date +%s)
    python3 -m src.main ingest "$SOURCE" --config "$CONFIG"
    T1=$(date +%s)
    echo "    Done in $((T1 - T0))s"
    echo ""
else
    echo ">>> Step 1/3: Ingest — SKIPPED"
    echo ""
fi

# Step 2: Build graph
if [ "$SKIP_GRAPH" = false ]; then
    echo ">>> Step 2/3: Building knowledge graph..."
    T0=$(date +%s)
    python3 -m src.main build-graph --config "$CONFIG"
    T1=$(date +%s)
    echo "    Done in $((T1 - T0))s"
    echo ""
else
    echo ">>> Step 2/3: Build graph — SKIPPED"
    echo ""
fi

# Step 3: Build communities
if [ "$SKIP_COMMUNITIES" = false ]; then
    echo ">>> Step 3/3: Detecting communities & generating summaries..."
    T0=$(date +%s)
    python3 -m src.main build-communities --config "$CONFIG"
    T1=$(date +%s)
    echo "    Done in $((T1 - T0))s"
    echo ""
else
    echo ">>> Step 3/3: Build communities — SKIPPED"
    echo ""
fi

PIPELINE_END=$(date +%s)
echo "========================================"
echo "  Pipeline complete in $((PIPELINE_END - PIPELINE_START))s"
echo "  Artifacts:"
echo "    data/processed/chunks.json"
echo "    data/graphs/knowledge_graph.json"
echo "    data/graphs/communities.json"
echo "========================================"
