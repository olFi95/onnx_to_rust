# Experiment to compile onnx directly into rust projects

You will need the tool onnx2c, i have provided a compiled version. Build your own from [sources of their github page](https://github.com/kraiskil/onnx2c)  
Additionally you will need a .onnx file of your choice, put it in resources.  
The build.rs will generate the c code from the .onnx file and then compile the c code.  
Note that everything is quite static, so if you choose to change the .onnx you will have to change the names in the build.rs and adapt the "entry" signature so the input and output dimensions are correct. you can get them from the very end of the generated .c file in /src/generated (in case you don't know the network of the .onnx).