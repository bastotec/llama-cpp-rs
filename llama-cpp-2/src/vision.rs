//! # Vision Language Model Support
//!
//! This module provides convenient APIs for working with Vision Language Models (VLMs)
//! using the MTMD (Multimodal Text, Media, Data) backend from llama.cpp.

use std::path::Path;

use crate::context::LlamaContext;
use crate::context::params::LlamaContextParams;
use crate::llama_backend::LlamaBackend;
use crate::model::LlamaModel;
#[cfg(feature = "mtmd")]
use crate::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use crate::sampling::LlamaSampler;
use crate::llama_batch::LlamaBatch;
use crate::token::LlamaToken;
use thiserror::Error;

use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;

/// Errors that can occur during VLM operations.
#[derive(Debug, Error)]
pub enum VisionError {
    #[error("MTMD initialization failed: {0}")]
    MtmdInitError(String),

    #[error("Image load failed: {0}")]
    ImageLoadError(String),

    #[error("Tokenization failed: {0}")]
    TokenizationError(String),

    #[error("Encoding failed: {0}")]
    EncodingError(String),

    #[error("Generation failed: {0}")]
    GenerationError(String),

    #[error("No mmproj file provided for VLM model")]
    MissingMmproj,

    #[error("Model does not support vision input")]
    VisionNotSupported,
}

/// Configuration parameters for VLM context creation.
#[derive(Debug, Clone)]
pub struct VisionContextParams {
    /// Number of threads to use for processing.
    pub n_threads: i32,
    /// Whether to use GPU acceleration for the multimodal projector.
    pub use_gpu: bool,
    /// Print timing information.
    pub print_timings: bool,
    /// Path to the multimodal projector file.
    pub mmproj_path: Option<String>,
    /// Maximum number of tokens in context.
    pub n_tokens: NonZeroU32,
    /// Batch size for processing.
    pub batch_size: i32,
}

impl Default for VisionContextParams {
    fn default() -> Self {
        Self {
            n_threads: 4,
            use_gpu: true,
            print_timings: false,
            mmproj_path: None,
            n_tokens: NonZeroU32::new(4096).unwrap(),
            batch_size: 1,
        }
    }
}

impl From<VisionContextParams> for MtmdContextParams {
    fn from(params: VisionContextParams) -> Self {
        let media_marker = CString::new("<__media__>").unwrap();

        MtmdContextParams {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker,
        }
    }
}

/// Input for VLM inference combining image and text.
#[derive(Debug)]
pub enum VisionInput<'a> {
    Text { text: String },
    Image { path: &'a Path, text: String },
}

impl<'a> VisionInput<'a> {
    pub fn from_text(text: &str) -> Self {
        VisionInput::Text { text: text.to_string() }
    }

    pub fn from_image(path: &'a Path, text: &str) -> Self {
        VisionInput::Image { path, text: text.to_string() }
    }
}

/// A context for VLM inference with MTMD support.
#[derive(Debug)]
pub struct VisionContext<'a> {
    pub mtmd_ctx: MtmdContext,
    pub batch: LlamaBatch<'a>,
    pub bitmaps: Vec<MtmdBitmap>,
    pub n_past: i32,
    params: VisionContextParams,
}

impl<'a> VisionContext<'a> {
    /// Create a new VLM context from a model and the multimodal projector file.
    #[cfg(feature = "mtmd")]
    pub fn new(
        model: &'a LlamaModel,
        backend: &LlamaBackend,
        params: VisionContextParams,
    ) -> Result<Self, VisionError> {
        let mmproj_path = params.mmproj_path.as_ref()
            .ok_or(VisionError::MissingMmproj)?;

        let mtmd_params = MtmdContextParams::from(params.clone());
        let mtmd_ctx = MtmdContext::init_from_file(mmproj_path, model, &mtmd_params)
            .map_err(|e| VisionError::MtmdInitError(e.to_string()))?;

        if !mtmd_ctx.support_vision() {
            return Err(VisionError::VisionNotSupported);
        }

        let batch = LlamaBatch::new(params.n_tokens.get() as usize, 1);

        Ok(Self {
            mtmd_ctx,
            batch,
            bitmaps: Vec::new(),
            n_past: 0,
            params,
        })
    }

    /// Load an image from a file path.
    #[cfg(feature = "mtmd")]
    pub fn load_image(&mut self, path: &Path) -> Result<(), VisionError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, path.to_str().unwrap())
            .map_err(|e| VisionError::ImageLoadError(e.to_string()))?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Run VLM inference with the given input.
    #[cfg(feature = "mtmd")]
    pub fn run(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        input: VisionInput,
    ) -> Result<String, VisionError> {
        match input {
            VisionInput::Text { text } => {
                let input_text = MtmdInputText {
                    text,
                    add_special: true,
                    parse_special: true,
                };
                let chunks = self.mtmd_ctx.tokenize(input_text, &[])
                    .map_err(|e| VisionError::TokenizationError(e.to_string()))?;

                self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, 0, 0, self.params.batch_size, true)
                    .map_err(|e| VisionError::EncodingError(e.to_string()))?;

                self.generate(model, context, -1)
            }
            VisionInput::Image { path, text } => {
                self.load_image(path)?;

                let mut prompt = text.clone();
                let default_marker = crate::mtmd::mtmd_default_marker().to_string();
                if !prompt.contains(&default_marker) {
                    prompt.push_str(&default_marker);
                }

                let input_text = MtmdInputText {
                    text: prompt,
                    add_special: true,
                    parse_special: true,
                };

                let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();
                let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)
                    .map_err(|e| VisionError::TokenizationError(e.to_string()))?;

                self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, 0, 0, self.params.batch_size, true)
                    .map_err(|e| VisionError::EncodingError(e.to_string()))?;

                self.bitmaps.clear();

                self.generate(model, context, -1)
            }
        }
    }

    /// Generate text from the model.
    #[cfg(feature = "mtmd")]
    fn generate(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        n_predict: i32,
    ) -> Result<String, VisionError> {
        let mut sampler = LlamaSampler::greedy();
        let mut output = String::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };

        for _i in 0..max_predict {
            let token = sampler.sample(context, -1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model.token_to_str(token, crate::model::Special::Tokenize)
                .map_err(|e| VisionError::GenerationError(e.to_string()))?;
            output.push_str(&piece);

            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)
                .map_err(|e| VisionError::GenerationError(e.to_string()))?;
            self.n_past += 1;

            context.decode(&mut self.batch)
                .map_err(|e| VisionError::EncodingError(e.to_string()))?;
        }

        Ok(output)
    }
}

impl LlamaModel {
    /// Create a new VLM context with the multimodal projector.
    #[cfg(feature = "mtmd")]
    pub fn new_vision_context(
        &self,
        backend: &LlamaBackend,
        params: VisionContextParams,
    ) -> Result<VisionContext<'_>, VisionError> {
        VisionContext::new(self, backend, params)
    }

    /// Create a new LlamaContext for use with VLM inference.
    #[cfg(feature = "mtmd")]
    pub fn new_vision_llama_context(
        &self,
        backend: &LlamaBackend,
        params: &VisionContextParams,
    ) -> Result<LlamaContext<'_>, crate::LlamaContextLoadError> {
        let context_params = LlamaContextParams::default()
            .with_n_threads(params.n_threads)
            .with_n_batch(params.batch_size.try_into().unwrap())
            .with_n_ctx(Some(params.n_tokens));
        self.new_context(backend, context_params)
    }
}
