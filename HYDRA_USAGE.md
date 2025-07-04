bash# 1. Run with default configuration
python scripts/train.py

# 2. Override specific parameters
python scripts/train.py train.batch_size=64 train.epochs=50

# 3. Use different model
python scripts/train.py model=efficientnet

# 4. Use different experiment preset
python scripts/train.py experiment=resnet_experiment

# 5. Quick test with 1 epoch
python scripts/train.py experiment=quick_test

# 6. Complex override example
python scripts/train.py model=efficientnet optimizer=adam scheduler=cosine train.batch_size=16 data.image_size=256

# 7. Multiple experiments in sequence
python scripts/train.py experiment=resnet_experiment --multirun
python scripts/train.py model=resnet50,efficientnet --multirun


