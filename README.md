# Explicit generalization training (WIP)

## Run instructions

Assuming you installed Python 3 with *tensorflow*, *scikit-learn*, *numpy* you need to:  

- Clone the project

```bash
git clone https://github.com/PaulEmmanuelSotir/generalization_loss.git
cd ./explicit_generalization_training
```

- Train the model on CIFAR dataset

```bash
python train_dnn.py
```

- Apply explicit generalization training on trained model to learn errors on testset predictions

```bash
python generalization_training.py
```
