//! Reproduction showing how to encounter the bug

use clap::Parser;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaChatMessage, LlamaModel},
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to a model that uses the Meta Llama 3 template
    model_path: PathBuf,
}

fn main() {
    let Args { model_path } = Args::parse();

    let backend = LlamaBackend::init().expect("backend failed to init");

    let params = LlamaModelParams::default();

    let model = LlamaModel::load_from_file(&backend, &model_path, &params)
        .unwrap_or_else(|e| panic!("model failed to load from {model_path:?}: {e:?}"));

    let chat = vec![
        LlamaChatMessage::new("system".into(), "You're a helpful assistant.".into())
            .expect("failed to make a system message"),
        LlamaChatMessage::new("user".into(), "What color is the sky?".into())
            .expect("failed to make a user message"),
        LlamaChatMessage::new("assistant".into(), "The sky is blue.".into())
            .expect("failed to make an assistant message"),
    ];

    let applied_template = model.apply_chat_template(None, chat.clone(), false);
    // PANICS: called `Result::unwrap()` on an `Err` value: BuffSizeError
    let applied_template = applied_template.unwrap();
    // (this assert_eq! here to ensure that a Llama-3-template-using model was passed)
    assert_eq!(applied_template, "<|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat color is the sky?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe sky is blue.<|eot_id|>");

    // Also including when add_ass is true,
    // because whether it's true or false doesn't affect (here at least)
    // that the buffer size too small error is returned

    let applied_template_ass = model.apply_chat_template(None, chat, true);
    // PANICS: called `Result::unwrap()` on an `Err` value: BuffSizeError
    let applied_template_ass = applied_template_ass.unwrap();
    // (this assert_eq! here to ensure that a Llama-3-template-using model was passed)
    assert_eq!(applied_template_ass, "<|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat color is the sky?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe sky is blue.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
}
