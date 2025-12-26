//! VLM-related tests for llama-cpp-2
//!
//! These tests verify VLM detection and basic functionality.
//! Note: Full VLM inference tests require actual model files.

#[cfg(test)]
mod model_vlm_detection {
    #[test]
    fn test_vlm_detection_methods_exist() {
        // These are compile-time tests to ensure the methods exist
        // They will only run if we have actual model files
        // For now, we just verify the methods compile
        let _model_params = llama_cpp_2::model::params::LlamaModelParams::default();
        let _context_params = llama_cpp_2::context::params::LlamaContextParams::default();
    }

    #[test]
    fn test_rope_type_vision() {
        use llama_cpp_2::model::RopeType;

        // Verify Vision rope type exists and is different from others
        assert_ne!(RopeType::Vision, RopeType::Norm);
        assert_ne!(RopeType::Vision, RopeType::NeoX);
        assert_ne!(RopeType::Vision, RopeType::MRope);
    }
}

#[cfg(test)]
mod vision_context_params {
    use llama_cpp_2::vision::VisionContextParams;

    #[test]
    fn test_default_params() {
        let params = VisionContextParams::default();

        assert_eq!(params.n_threads, 4);
        assert!(params.use_gpu);
        assert!(!params.print_timings);
        assert!(params.mmproj_path.is_none());
    }

    #[test]
    fn test_params_with_mmproj() {
        let params = VisionContextParams {
            mmproj_path: Some("test-mmproj.gguf".to_string()),
            n_threads: 8,
            use_gpu: false,
            ..Default::default()
        };

        assert_eq!(params.n_threads, 8);
        assert!(!params.use_gpu);
        assert_eq!(params.mmproj_path, Some("test-mmproj.gguf".to_string()));
    }
}

#[cfg(test)]
mod vision_input {
    use llama_cpp_2::vision::VisionInput;
    use std::path::Path;

    #[test]
    fn test_text_input() {
        let input = VisionInput::from_text("Hello, world!");
        match input {
            VisionInput::Text { text, .. } => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_image_path_input() {
        let path = Path::new("test.jpg");
        let input = VisionInput::from_image(path, "Describe this");
        match input {
            VisionInput::Image { path: p, text, .. } => {
                assert_eq!(p, Path::new("test.jpg"));
                assert_eq!(text, "Describe this");
            }
            _ => panic!("Expected Image variant"),
        }
    }
}
