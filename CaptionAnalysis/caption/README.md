### To use minicpm caption

    cd CaptionAnalysis/caption/minicpm
    conda env create -f environment.yml
    conda activate minicpm
    python minicpm_caption.py --src_dir path/to/frames --out_dir output/path

### To use llava caption
    
    cd CaptionAnalysis/caption/llava
    conda env create -f environment.yml
    conda activate llava_caption
    ollama pull llava #if not downloaded
    python llava_caption.py --src_dir path/to/frames --out_dir output/path
    
    