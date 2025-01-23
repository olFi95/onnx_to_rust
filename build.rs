use std::fs::File;
use std::io::Write;
use std::process::{Command, Stdio};
use cc;
fn main() {
    let onnx2c_status = Command::new("./onnx2c")
        .args(["resources/mnist-8.onnx"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to generate c code from resources/mnist-8.onnx")
        .wait_with_output()
        .expect("failed to wait on onnx2c");

    if !onnx2c_status.status.success(){
        panic!("onnx2c failed with non-zero status code");
    }

    let mut resnet_18_out = File::create("src/generated/mnist-8.c")
        .expect("Could not open src/generated/mnist-8.c");
    resnet_18_out.write_all(&onnx2c_status.stdout)
        .expect("Could not write to src/generated/mnist-8.c");

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("Compile generated Neuronal Network");
    // Use the `cc` crate to build a C file and statically link it.
    cc::Build::new()
        .file("src/generated/mnist-8.c")
        .opt_level(2) // Set optimisation level, 2 is a good compromise.
        .flag("-Os") // optimise for binary size.
        .compile("mnist-8");
}