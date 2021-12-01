#!/bin/bash
make_opts=(
    "WITH_NC=False WITH_O=True CHUNKER=enumerated"
    "WITH_NC=False WITH_O=True CHUNKER=spacy_np"
    "WITH_NC=True WITH_O=False CHUNKER=spacy_np"
    "WITH_NC=True WITH_O=True CHUNKER=spacy_np"
)
for MAKEOPTS in "${make_opts[@]}"; do
    echo "MAKEOPTS: ${MAKEOPTS}"
    eval ${MAKEOPTS} make all -j$(nproc)
done
# WITH_NC=True make all -j$(nproc)
# WITH_NC=False make all -j$(nproc)