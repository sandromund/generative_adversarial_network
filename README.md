# Generative Adversarial Network
Generative adversarial networks are machine learning systems that can learn to mimic a given distribution of data.
Learn more about it [here](https://realpython.com/generative-adversarial-networks/).


Training
```shell
 python src/main.py train --data data/climbs --batch 100 --epochs 10
```

Generate
```shell
python src/main.py generate --model runs:/3c6898b479dc4928bf2baa5133295a50/generator
```
Tracking
MlFlow Server
```shell
mlflow server --host 127.0.0.1 --port 8080
```
<!---
# Bouldern
https://kilterboard.app/
-->

### GPU Setup
See https://pytorch.org/get-started/locally/ 
````shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````