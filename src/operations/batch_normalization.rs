use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct BatchNormalizationOperation;

impl OperationCodeGenerator for BatchNormalizationOperation {
    /// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement BatchNormalization operation
            // Normalize activations using batch statistics
            unimplemented!("BatchNormalization operation not yet implemented")
        }
    }
}

