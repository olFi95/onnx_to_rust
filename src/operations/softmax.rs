use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct SoftmaxOperation;

impl OperationCodeGenerator for SoftmaxOperation {
    /// https://onnx.ai/onnx/operators/onnx__Softmax.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Softmax operation
            // Apply softmax normalization: exp(x) / sum(exp(x))
            unimplemented!("Softmax operation not yet implemented")
        }
    }
}

