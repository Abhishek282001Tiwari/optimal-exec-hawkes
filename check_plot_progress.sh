#!/bin/bash
echo "Checking plot generation progress..."
if [ -d "docs/paper_figures" ]; then
    echo "Figures directory exists."
    count=$(ls -1 docs/paper_figures/*.pdf 2>/dev/null | wc -l)
    echo "Number of PDF figures generated: $count"
    if [ $count -gt 0 ]; then
        echo "Generated figures:"
        ls -la docs/paper_figures/*.pdf
    else
        echo "No PDF figures yet - still generating..."
    fi
else
    echo "Figures directory not created yet - script is starting..."
fi
