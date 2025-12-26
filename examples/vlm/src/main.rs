//! # VLM Example
//!
//! This example demonstrates how to use Vision Language Models (VLMs) with llama-cpp-2.

use clap::Parser;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::vision::{VisionContextParams, VisionInput};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

/// VLM CLI tool using llama-cpp-2
#[derive(Parser, Debug)]
#[command(name = "vlm")]
#[command(about = "VLM CLI using llama-cpp-2", long_about = None)]
struct Args {
    /// Path to the model file (.gguf format)
    #[arg(long)]
    model: PathBuf,

    /// Path to the multimodal projector file (.gguf format)
    #[arg(long)]
    mmproj: PathBuf,

    /// Path to the input image
    #[arg(short, long)]
    image: PathBuf,

    /// Prompt to send to the VLM
    #[arg(short, long, default_value = "Describe this image in detail.")]
    prompt: String,

    /// Number of threads
    #[arg(long, default_value = "4")]
    threads: i32,

    /// Maximum number of tokens in context
    #[arg(long, default_value = "4096")]
    n_tokens: u32,

    /// Number of tokens to predict (-1 for unlimited)
    #[arg(long, default_value = "-1")]
    n_predict: i32,

    /// Batch size
    #[arg(long, default_value = "1")]
    batch_size: i32,

    /// Disable GPU acceleration
    #[arg(long)]
    no_gpu: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    println!("VLM Example - llama-cpp-2");
    println!("========================\n");

    // Check files exist
    if !args.model.exists() {
        eprintln!("Error: Model file not found: {:?}", args.model);
        std::process::exit(1);
    }
    if !args.mmproj.exists() {
        eprintln!("Error: mmproj file not found: {:?}", args.mmproj);
        std::process::exit(1);
    }
    if !args.image.exists() {
        eprintln!("Error: Image file not found: {:?}", args.image);
        std::process::exit(1);
    }

    // Initialize backend
    let backend = LlamaBackend::init()?;
    println!("Backend initialized");

    // Load model
    let mut model_params = LlamaModelParams::default();
    if !args.no_gpu {
        model_params = model_params.with_n_gpu_layers(99);
    }

    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)?;
    println!("Model loaded: {:?}", args.model.file_name().unwrap_or_default());

    // Check if it's a VLM
    if model.is_vision_model() {
        println!("VLM detected:");
        println!("  - Has encoder: {}", model.has_encoder());
        println!("  - Has decoder: {}", model.has_decoder());
    } else {
        println!("Note: Model doesn't report as VLM, but will try anyway");
    }

    // Create VLM context
    let vision_params = VisionContextParams {
        mmproj_path: Some(args.mmproj.to_string_lossy().to_string()),
        n_threads: args.threads,
        use_gpu: !args.no_gpu,
        print_timings: false,
        n_tokens: std::num::NonZeroU32::new(args.n_tokens).unwrap(),
        batch_size: args.batch_size,
        ..Default::default()
    };

    let mut vlm_ctx = model.new_vision_context(&backend, vision_params.clone())?;
    println!("VLM context created");

    // Create LlamaContext for generation
    let mut context = model.new_vision_llama_context(&backend, &vision_params)?;
    println!("LlamaContext created");

    // Run inference
    println!("\nProcessing image: {:?}", args.image);
    println!("Prompt: {}\n", args.prompt);
    println!("Generating response...\n");

    let input = VisionInput::from_image(&args.image, &args.prompt);
    let response = vlm_ctx.run(&model, &mut context, input)?;

    println!("\n\nResponse: {}", response);

    Ok(())
}
