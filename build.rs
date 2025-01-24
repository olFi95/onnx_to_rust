use std::fs::File;
use std::io::Write;
use std::process::{Command, Stdio};
// use cc;
fn main() {
    prost_build::compile_protos(&["resources/onnx.proto"], &["resources"])
        .expect("Failed to compile onnx.proto");
}