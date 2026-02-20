use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct MaxPoolOperation;

impl OperationCodeGenerator for MaxPoolOperation {
    /// https://onnx.ai/onnx/operators/onnx__MaxPool.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement MaxPool (Max Pooling) operation
            // Apply max pooling with kernel size and stride
            unimplemented!("MaxPool operation not yet implemented")
        }
    }
}

