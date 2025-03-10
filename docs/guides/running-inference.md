!!! note
    In this section, we expect that you have already installed NOS and have already [started the server](./starting-the-server.md).

```python linenums="1"
import nos

nos.init(runtime="auto")
```

## Connecting to the NOS Server

You can now send inference requests using the NOS client.
Let's start by importing the NOS client and creating an `Client` instance. The client instance is used to send inference requests to the NOS server via gRPC.

```python
from nos.client import Client, TaskType

# Create a client that connects to the server via gRPC (over 50051)
client = Client("[::]:50051")

# We provide helper functions to wait for the server to be ready
# if the server is simultaneously spun up in a separate process.
client.WaitForServer()

# Finally, we can check if the server is healthy.
client.IsHealthy()
```

## Running Inference with the `client.Module` Interface

NOS provides a `client.Module` interface to get model handles for remote-model execution. Let's see an example of running `yolox/nano` to run 2D object detection on a sample image.

```python
# Get a model handle for yolox/nano
detect2d = client.Module("yolox/nano")

# Run inference on a sample image
img = Image.open("sample.jpg")
predictions = detect2d(images=[img])
```

In essense, the `client.Module` is an inference [`Module`](../api/client.md#nosclientgrpcclient)  that provides a *logical* handle for the model on the remote server. The model handle could contain multiple replicas, or live in a specialized runtime (GPU, ASICs), however, the user does not need to be aware of these abstractions. Instead, you can simply call the model as a regular Python function where the task gets dispatched to the associated set of remote workers.

## More examples

### Text-to-image generation with [StableDiffusionV2](https://huggingface.co/stabilityai/stable-diffusion-2)

```python
prompts = ["fox jumped over the moon", "fox jumped over the sun"]
sdv2 = client.Module("stabilityai/stable-diffusion-2")
images = sdv2(inputs={
    "prompts": prompts, "width": 512, "height": 512, "num_images": 1
})
images[0]
```

### Image-embedding with [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32)

```python
clip = client.Module("openai/clip")
predictions = clip.encode_image(inputs={"images": [img]})
predictions["embedding"].shape
```

### Text-embedding with [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
```python
clip = client.Run("openai/clip")
predictions = clip.encode_text(inputs={
    "texts": ["fox jumped over the mooon", "fox jumped over the sun"]
})
predictions["embedding"].shape
```
