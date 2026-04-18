# Jossnet-bitnet

## Introduction 

This project is an implementation of the [bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) model from Microsoft for Apple Silicon using Metal. 

First of all, this project is an opportunity for me to learn more about how an LLM works, as well as to deepen my understanding of GPU programming with Metal.

## Work in progress

This project is still in progress. The model is able to return logical results, but it is currently very slow and not accurate.

### Current performance
* M3 chip (8 CPU - 10 GPU | 16 GB) ~ 0.84s per token 

### Next steps
* Work on optimization
    * Add cache for k and v buffers
* Add a real tokenizer
* Write a report on this project

## Model

This project uses the [bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) model. By default, Microsoft provides the model in `.safetensors` format, so I used Python to convert it into `.bin` files. 

The converted model is available on my [Hugging Face repository](https://huggingface.co/Dr-joss/bitnet-b1.58-2B-metal-weight).

## License & References

* Hugging Face model: [Dr-joss/bitnet-b1.58-2B-metal-weight](https://huggingface.co/Dr-joss/bitnet-b1.58-2B-metal-weight)
* License: MIT License (Copyright © FIGUEIRAS Jossua)
