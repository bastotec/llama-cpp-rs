# VLM Support for llama-cpp-2 (MTMD-based Implementation)

This document describes the current VLM (Vision Language Model) support in llama-cpp-2.

## Important: API Change Notice

The previous research documented in this file referenced an **obsolete API**. The current implementation uses **MTMD (Multimodal Text, Media, Data)** library from llama.cpp, which is the standard multimodal interface.

**The `llama_image_*` functions documented elsewhere do not exist in current llama.cpp.**

## Architecture

### MTMD Library
llama.cpp's multimodal support is provided by the `libmtmd` library, which handles:
- Image loading and preprocessing
- Audio loading and preprocessing
- Tokenization of multimodal inputs
- Encoding of image/audio to embeddings

### llama-cpp-2 Integration
The Rust bindings provide:
1. **Direct MTMD bindings** (`llama-cpp-2/src/mtmd.rs`) - Low-level access to MTMD functions
2. **Vision convenience module** (`llama-cpp-2/src/vision.rs`) - High-level VLM API

## Implementation Status

### Completed

1. **VLM Detection Methods** (in `LlamaModel`):
   - `has_encoder()` - Check if model has a vision encoder
   - `has_decoder()` - Check if model has a decoder
   - `is_vision_model()` - Convenience method combining both checks
   - `is_hybrid()` - Check if model is hybrid (Jamba, Granite, etc.)
   - `is_diffusion()` - Check if model is diffusion-based

2. **MTMD Bindings** (already existed):
   - `MtmdContext` - Multimodal context for VLM inference
   - `MtmdBitmap` - Image/audio data wrapper
   - `MtmdInputChunks` - Tokenized input chunks
   - `MtmdInputText` - Text input configuration
   - Full access to `mtmd_*` functions via `llama_cpp_sys_2`

3. **Vision Module** (`llama-cpp-2/src/vision.rs`):
   - `VisionContext` - High-level VLM context
   - `VisionContextParams` - VLM-specific configuration
   - `VisionInput` - Input combining images and text
   - `VisionImageInput` - Image source (path or bitmap)

### In Progress

- VLM Example (`examples/vlm/src/main.rs`)
- Comprehensive VLM tests

## Usage

### Basic VLM Setup

```rust
use llama_cpp_2::vision::{VisionContextParams, VisionInput};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = llama_cpp_2::LlamaBackend::init()?;
    
    let model = LlamaModel::load_from_file(
        &backend,
        Path::new("models/gemma-3-4b-it-Q4_K_M.gguf"),
        Default::default(),
    )?;
    
    // Check if VLM
    assert!(model.is_vision_model());
    
    // Create VLM context
    let mut ctx = model.new_vision_context(
        &backend,
        VisionContextParams {
            mmproj_path: Some("models/gemma-3-4b-it-mmproj-Q4_K_M.gguf".to_string()),
            ..Default::default()
        },
    )?;
    
    // Run inference
    let response = ctx.run(
        &model,
        VisionInput::from_image_path(Path::new("image.jpg"), "Describe this image."),
    )?;
    
    println!("Response: {}", response);
    Ok(())
}
```

### Using MTMD Directly

For more control, use the MTMD module directly:

```rust
use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams, MtmdInputText};

let ctx_params = MtmdContextParams::default();
let mtmd_ctx = MtmdContext::init_from_file(
    "mmproj.gguf",
    &model,
    &ctx_params,
)?;

let bitmap = MtmdBitmap::from_file(&mtmd_ctx, "image.jpg")?;

let text = MtmdInputText {
    text: "Describe this image: <__media__>",
    add_special: true,
    parse_special: true,
};

let chunks = mtmd_ctx.tokenize(text, &[&bitmap])?;
```

## Supported Models

| Model | Architecture | Status |
|-------|-------------|--------|
| Gemma 3 | Encoder-Decoder | Supported |
| SmolVLM | Encoder-Decoder | Supported |
| Pixtral | Encoder-Decoder | Supported |
| Qwen2-VL | Encoder-Decoder | Supported |
| Qwen2.5-VL | Encoder-Decoder | Supported |
| InternVL 2.5/3 | Encoder-Decoder | Supported |
| Llama 4 Scout | Encoder-Decoder | Supported |
| Moondream2 | Encoder-Decoder | Supported |

## Model Files

VLM models typically require:
1. **Main model file** (`.gguf`) - The language model weights
2. **Multimedia projector file** (`mmproj.gguf`) - The vision encoder weights

For HuggingFace models, the mmproj file is often distributed alongside the main model with "mmproj" in the filename.

## Feature Flags

- `mtmd` - Enable MTMD and VLM support (required)
- `metal` - Metal GPU acceleration (macOS)
- `cuda` - CUDA GPU acceleration (NVIDIA)
- `vulkan` - Vulkan GPU acceleration

## Example

See `examples/vlm/src/main.rs` for a complete CLI example:

```bash
cargo run --example vlm --features mtmd,metal -- \
    --model model.gguf \
    --mmproj mmproj.gguf \
    --image photo.jpg \
    --prompt "Describe this image."
```

## References

- [llama.cpp Multimodal Documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/multimodal.md)
- [MTMD Header](https://github.com/ggerganov/llama.cpp/blob/master/tools/mtmd/mtmd.h)
- [HuggingFace GGUF Models with Vision](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc)
