# VLM Quick Reference

## Feature Flags

```toml
[dependencies]
llama-cpp-2 = { path = "llama-cpp-2", features = ["mtmd", "sampler"] }
```

Required features:
- `mtmd` - Enable multimodal (VLM) support

Optional features:
- `metal` - Metal GPU acceleration (macOS)
- `cuda` - CUDA GPU acceleration
- `vulkan` - Vulkan GPU acceleration

## VLM Detection

```rust
use llama_cpp_2::LlamaModel;

let model = LlamaModel::load_from_file(&backend, "model.gguf", Default::default())?;

model.is_vision_model()  // bool - Is this a VLM?
model.has_encoder()      // bool - Has vision encoder?
model.has_decoder()      // bool - Has text decoder?
```

## Basic VLM Usage

```rust
use llama_cpp_2::vision::{VisionContextParams, VisionInput};
use std::path::Path;

let backend = LlamaBackend::init()?;
let model = LlamaModel::load_from_file(&backend, "model.gguf", Default::default())?;

let mut ctx = model.new_vision_context(
    &backend,
    VisionContextParams {
        mmproj_path: Some("mmproj.gguf".to_string()),
        ..Default::default()
    },
)?;

let response = ctx.run(
    &model,
    VisionInput::from_image_path(Path::new("image.jpg"), "Describe this image."),
)?;
```

## VisionContextParams

```rust
VisionContextParams {
    n_threads: 4,                    // Processing threads
    use_gpu: true,                   // Use GPU for projector
    print_timings: false,            // Print timing info
    image_max_tokens: None,          // Max tokens for dynamic resolution
    image_min_tokens: None,          // Min tokens for dynamic resolution
    mmproj_path: None,               // Path to mmproj file
}
```

## VisionInput Types

```rust
// Text only
VisionInput::from_text("Hello, world!")

// Image from file path
VisionInput::from_image_path(Path::new("image.jpg"), "Describe this image.")

// Image from bitmap
VisionInput::from_image_bitmap(&bitmap, "What's in this image?")

// Multiple images (not yet implemented)
VisionInput::from_images_and_text(vec![img1, img2], "Compare these images.")
```

## CLI Example

```bash
cargo run --example vlm --features mtmd,metal -- \
    --model model.gguf \
    --mmproj mmproj.gguf \
    --image photo.jpg \
    --prompt "Describe this image." \
    --threads 4 \
    --max-tokens 512
```

## Model Files

VLM models require two GGUF files:

| File | Description |
|------|-------------|
| `model.gguf` | Main LLM weights |
| `mmproj.gguf` | Vision encoder weights |

The mmproj file is typically found on HuggingFace with "mmproj" in the name.

## Supported Models

| Model | Size | Notes |
|-------|------|-------|
| Gemma 3 | 4B, 12B, 27B | Encoder-decoder |
| SmolVLM | 256M-2.2B | Small, efficient |
| Qwen2-VL | 2B, 7B | High quality |
| Qwen2.5-VL | 3B-72B | Latest Qwen VLM |
| Pixtral | 12B | Mistral's VLM |
| InternVL | 1B-14B | Strong vision |
| Llama 4 Scout | 17B | Meta's latest |
| Moondream2 | ~1.5B | Very small |

## Using MTMD Directly

```rust
use llama_cpp_2::mtmd::{MtmdContext, MtmdBitmap, MtmdContextParams};

let ctx_params = MtmdContextParams::default();
let mtmd_ctx = MtmdContext::init_from_file("mmproj.gguf", &model, &ctx_params)?;

let bitmap = MtmdBitmap::from_file(&mtmd_ctx, "image.jpg")?;
let support_vision = mtmd_ctx.support_vision();
let decode_mrope = mtmd_ctx.decode_use_mrope();
```

## Error Handling

```rust
use llama_cpp_2::vision::VisionError;

match ctx.run(&model, input) {
    Ok(response) => println!("{}", response),
    Err(VisionError::NotAVisionModel) => {
        eprintln!("Model doesn't support vision!");
    }
    Err(VisionError::MissingMmproj) => {
        eprintln!("Please provide --mmproj argument");
    }
    Err(VisionError::ImageLoadError(e)) => {
        eprintln!("Failed to load image: {}", e);
    }
    Err(e) => {
        eprintln!("VLM error: {}", e);
    }
}
```

## Common Issues

### "Model is not a Vision Language Model"
- The model file may not be a VLM
- Check with `model.is_vision_model()`

### "No mmproj file provided"
- Encoder models need the multimodal projector
- Download from HuggingFace (search for "mmproj")

### "Model does not support vision input"
- The mmproj may be incompatible
- Verify both files are from the same model release

## Performance Tips

1. **Increase context size**: VLMs need more context (8192+)
2. **Use GPU offload**: Set `use_gpu: true` for mmproj
3. **Limit image tokens**: Use `image_max_tokens` for dynamic resolution models
4. **More threads**: Increase `n_threads` for faster encoding
