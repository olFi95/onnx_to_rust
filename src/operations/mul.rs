use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct MulOperation;

impl OperationCodeGenerator for MulOperation {
    /// https://onnx.ai/onnx/operators/onnx__Mul.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Mul (Multiplication) operation
            // Element-wise multiplication of two tensors
            unimplemented!("Mul operation not yet implemented")
        }
    }
}

