# optimize-deploy



### example

#### Optimze 
```bash
python src/convert.py --model_path examples/resnet50/ --optimized_model_path nebullvm_optimized --triton_model_path triton_models --task image_classification
```

#### Deploy
```bash
# Build
docker build -t tritonserver -f deploy/Dockerfile .
# Run
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models tritonserver \
  tritonserver --model-repository=/models
```

#### Infer
```bash
python clients/img_client.py --model_name ResNet50_pipeline --image clients/sample_inputs/mug.jpg 
```