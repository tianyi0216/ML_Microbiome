#!/bin/bash

# Iterate over each sample directory
for sample_dir in */; do
    echo "Processing sample: $sample_dir"
    cd "$sample_dir"
    
    # Decompress each .gz file
    for gz_file in *.gz; do
        echo "Decompressing $gz_file"
        gunzip "$gz_file"
    done
    
    # Go back to the main 'data' directory to process the next sample
    cd ..
done

echo "Decompression complete!"






