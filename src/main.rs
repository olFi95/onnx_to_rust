
use libc::{c_float, c_void};

// Declare the C function using extern block
extern "C" {
    ///
    /// input has shape: [1][28][28]
    /// output has shape: [1][10]
    fn entry(tensor_input: *const c_float, tensor_output: *mut c_float);
}

fn main() {
    let mut tensor_input: Vec<c_float> = vec![0.0; 1*28*28];

    let mut tensor_output: Vec<c_float> = vec![0.0; 1*10];

    unsafe {
        entry(tensor_input.as_ptr(), tensor_output.as_mut_ptr());
    }

    // Print the output tensor to verify
    println!("{:?}", tensor_output);}
