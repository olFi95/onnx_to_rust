use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct ConvOperation;

impl OperationCodeGenerator for ConvOperation {
    /// https://onnx.ai/onnx/operators/onnx__Conv.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Conv (Convolution) operation
            // Convolutional layer with kernel, stride, padding
            unimplemented!("Conv operation not yet implemented")
        }
    }
}

