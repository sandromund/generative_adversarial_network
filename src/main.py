import click

from generate import generate_sample_mlflow
from train import train_models


@click.group()
def cli():
    pass


@click.command()
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--epochs', default=300, help='Number of training epochs')
@click.option('--batch', default=32, help='Training batch size')
def train(lr, epochs, batch, save):
    train_models(batch_size=batch, lr=lr, num_epochs=epochs)


@click.command()
@click.option('--model', help='MlFlow model uri')
def generate(model):
    generate_sample_mlflow(model_uri=model)


cli.add_command(train)
cli.add_command(generate)

if __name__ == '__main__':
    cli()
