class Config:
    # Title to be used in the Streamlit app
    page_title = ""

    # Placeholder model names — replace these with your actual models
    ollama_models = [
        "llama3",  # LLaMA model for caption generation
        "llava"    # LLAVA model for image description
    ]
