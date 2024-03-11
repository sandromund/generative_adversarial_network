import click

from generate import generate_sample
from train import train_models


@click.group()
def cli():
    pass


@click.command()
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--epochs', default=300, help='Number of training epochs')
@click.option('--batch', default=32, help='Training batch size')
@click.option('--save', default="generator.pt", help='Model save path')
def train(lr, epochs, batch, save):
    train_models(batch_size=batch, model_save_path=save, lr=lr, num_epochs=epochs, plot_training=True)


@click.command()
@click.option('--model', default="generator.pt", help='Generator model to load')
def generate(model):
    generate_sample(model_path=model)


cli.add_command(train)
cli.add_command(generate)

if __name__ == '__main__':
    cli()
