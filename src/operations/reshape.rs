use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct ReshapeOperation;

impl OperationCodeGenerator for ReshapeOperation {
    /// https://onnx.ai/onnx/operators/onnx__Reshape.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Reshape operation
            // Reshape tensor to new dimensions
            unimplemented!("Reshape operation not yet implemented")
        }
    }
}

