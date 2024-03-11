# Generative Adversarial Network
Generative adversarial networks are machine learning systems that can learn to mimic a given distribution of data.
Learn more about it [here](https://realpython.com/generative-adversarial-networks/).


Training
```shell
 python src/main.py train --epochs 100
```

Generate
```shell
python src/main.py generate --model generator.pt
```
Tracking
MlFlow Server on http://localhost:5000
```shell
mlflow server --host 127.0.0.1 --port 8080

```