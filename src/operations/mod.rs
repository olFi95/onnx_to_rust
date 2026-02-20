// Module for ONNX operation code generators
// Each operation has its own file with specific code generation logic

pub mod add;
pub mod conv;
pub mod relu;
pub mod maxpool;
pub mod reshape;
pub mod matmul;
pub mod gemm;
pub mod softmax;
pub mod mul;
pub mod flatten;
pub mod batch_normalization;
pub mod dropout;

use quote::__private::TokenStream;

/// Trait for operation-specific code generation
pub trait OperationCodeGenerator {
    /// Generate the implementation code for this operation
    fn generate_implementation(&self, inputs: &[String], outputs: &[String]) -> TokenStream;
}

