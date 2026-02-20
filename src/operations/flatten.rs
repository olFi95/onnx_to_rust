use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct FlattenOperation;

impl OperationCodeGenerator for FlattenOperation {
    /// https://onnx.ai/onnx/operators/onnx__Flatten.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Flatten operation
            // Flatten tensor to a 2D matrix
            unimplemented!("Flatten operation not yet implemented")
        }
    }
}

