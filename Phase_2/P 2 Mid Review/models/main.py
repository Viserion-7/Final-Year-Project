def main():
    print("Hello from models!")
    # uv run python - << 'EOF'
    import torch
    import transformers
    print("Torch version:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Transformers version:", transformers.__version__)



if __name__ == "__main__":
    main()
