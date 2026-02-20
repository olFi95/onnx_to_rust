use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct DropoutOperation;

impl OperationCodeGenerator for DropoutOperation {
    /// https://onnx.ai/onnx/operators/onnx__Dropout.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Dropout operation
            // Randomly zero out elements during training
            unimplemented!("Dropout operation not yet implemented")
        }
    }
}

