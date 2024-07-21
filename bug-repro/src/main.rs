//! Reproduction showing how to encounter the bug

use clap::Parser;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to any model
    model_path: PathBuf,
}

fn main() {
    let Args { model_path } = Args::parse();

    let backend = LlamaBackend::init().expect("backend failed to init");

    let params = LlamaModelParams::default();

    let model = LlamaModel::load_from_file(&backend, &model_path, &params)
        .unwrap_or_else(|e| panic!("model failed to load from {model_path:?}: {e:?}"));

    // anything sufficiently small
    let buf_size = 15;
    // PANICS:
    // thread 'main' panicked at llama-cpp-2\src\model.rs:384:13:
    // assertion `left == right` failed: llama.cpp guarantees that the returned int 344 is the length of the string 14 but that was not the case
    //  left: 344
    // right: 14
    //
    // (344 represents the length of the template and varies per model / fine-tune)
    let chat_template = model.get_chat_template(buf_size);
}
