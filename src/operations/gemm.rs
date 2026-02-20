use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct GemmOperation;

impl OperationCodeGenerator for GemmOperation {
    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Gemm (General Matrix Multiplication) operation
            // Compute Y = alpha * A * B + beta * C
            unimplemented!("Gemm operation not yet implemented")
        }
    }
}

