#!/usr/bin/env bash

# Create assets directory structure
mkdir -p docs/assets/img/sample-analysis

# Copy key analysis plots (when they exist)
if [ -d "analysis/sample-output/plots" ]; then
    cp analysis/sample-output/plots/batch_size_scaling.png docs/assets/img/sample-analysis/
    cp analysis/sample-output/plots/batch_size_scaling_by_memory.png docs/assets/img/sample-analysis/
    cp analysis/sample-output/plots/latency_analysis.png docs/assets/img/sample-analysis/
    cp analysis/sample-output/plots/memory_efficiency.png docs/assets/img/sample-analysis/
    cp analysis/sample-output/plots/throughput_comparison.png docs/assets/img/sample-analysis/
fi

echo "✅ README assets updated"