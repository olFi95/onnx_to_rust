use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct MatMulOperation;

impl OperationCodeGenerator for MatMulOperation {
    /// https://onnx.ai/onnx/operators/onnx__MatMul.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement MatMul (Matrix Multiplication) operation
            // Perform matrix multiplication of two tensors
            unimplemented!("MatMul operation not yet implemented")
        }
    }
}

