## TensorRT使用

#### 一、TensorRT的主要功能：

![image-20210706133708865](https://user-images.githubusercontent.com/13610075/124576817-4254d200-de7f-11eb-989a-c2aed33f0f3c.png)


为深度学习推理应用提供低延迟和高**吞吐量**。

#### 二、TensorRT的用法：

1、Pytorch->ONNX

2、TensorRT环境准备

3、TensorRT Python API编程 

3、执行推理

##### 2.1、Pytorch->ONNX

```python
import torch.onnx as torch_onnx

model_onnx_path = "torch_model.onnx"
model = Model().cuda()
model.load_state_dict(torch.load("../weights/3s/SDM_weight_100.pkl"))
model.train(False)

# Export the model to an ONNX file
dummy_input = torch.randn(1, 3, 288, 144, requires_grad=True, device='cuda')
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
#print("Export of torch_model.onnx complete!")
```

##### 2.2、TensorRT环境准备

安装pycuda

```shell
pip install 'pycuda<2021.1'
```

<details>
    https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda
</details>


下载TensorRT包

```shell
cat /etc/issue					#查看系统版本
cat /usr/local/cuda/version.txt	#查看cuda版本
wget 包链接
#https://developer.nvidia.com/zh-cn/tensorrt
```

安装TensorRT（以wget tar包为例子）

```shell
#-----------------------Unpack the tar file.---------------------------
version="8.x.x.x"
arch=$(uname -m)
cuda="cuda-x.x"
cudnn="cudnn8.x"
tar xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.${cudnn}.tar.gz
#Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
#Install the Python TensorRT wheel file.
cd TensorRT-${version}/python
sudo pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl
#Install the Python UFF wheel file. This is only required if you plan to use TensorRT with TensorFlow.
cd TensorRT-${version}/uff
sudo pip3 install uff-0.6.9-py2.py3-none-any.whl
which convert-to-uff#Check the installation
#Install the Python graphsurgeon wheel file.
cd TensorRT-${version}/graphsurgeon
sudo pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl
#Install the Python onnx-graphsurgeon wheel file.
cd TensorRT-${version}/onnx_graphsurgeon
sudo pip3 install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```

<details>
    https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
</details>

##### 2.3、TensorRT Python API编程

```python
import tensorrt as trt
import pycuda.driver as cuda


G_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(G_LOGGER) as builder, builder.create_network() as network,trt.OnnxParser(network, G_LOGGER) as parser:
    builder.max_batch_size = args.batch_size
    #builder.max_workspace_size = 1 << 30
    with open(onnx_file_path, 'rb') as model:
    	parser.parse(model.read())
    engine = builder.build_cuda_engine(network)
    # 保存计划文件
    #print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
    #with open(args.engine_file_path, "wb") as f:
    #	f.write(engine.serialize())
    

# 准备输入输出数据
img = Image.open('XXX.jpg')
img = D.transform(img).unsqueeze(0)
img = img.numpy()
output = np.empty((1, 2), dtype=np.float32)

#创建上下文
context = engine.create_execution_context()

# 分配内存
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
bindings = [int(d_input), int(d_output)]

# pycuda操作缓冲区
stream = cuda.Stream()

# 将输入数据放入device
cuda.memcpy_htod_async(d_input, img, stream)

# 执行模型
context.execute_async(100, bindings, stream.handle, None)

# 将预测结果从从缓冲区取出
cuda.memcpy_dtoh_async(output, d_output, stream)
# 线程同步
stream.synchronize()

print(output)
```

<details>
    https://www.pythonheidong.com/blog/article/414249/147956bcff90e79e65e9/
</details>

