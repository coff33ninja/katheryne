from pathlib import Path
import os
import torch
import sys
from torch.utils.data import DataLoader  # Add this import
from models.decoders import HardwareManager

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.genshin_assistant import GenshinAssistantTrainer
from tqdm import tqdm


class GenshinAssistantTrainer:
    def __init__(self, dataset, batch_size, learning_rate, epochs, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.dataloader = self.create_dataloader()  # Add this line to initialize dataloader
        self.model = self.create_model()  # Add this line to initialize model
        self.vocab_size = 587  # Add this line to define vocab_size
        self.criterion = torch.nn.CrossEntropyLoss()  # Add this line to define criterion

    def create_dataloader(self):
        # Implement the method to create and return a dataloader
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def create_model(self):
        # Implement the method to create and return the model
        # Example:
        from models.genshin_model import GenshinModel
        return GenshinModel()


def load_dataset(project_root):
    # Implement the dataset loading logic here
    # For example, you can load a dataset from a file or directory
    # Return the loaded dataset
    # Example implementation:
    from torch.utils.data import TensorDataset
    import torch

    # Dummy dataset for illustration purposes
    data = torch.randn(100, 10)  # 100 samples, 10 features each
    targets = torch.randint(0, 2, (100,))  # 100 binary targets
    return TensorDataset(data, targets)

def main():
    # Get configuration from environment variables with sensible defaults
    epochs = int(os.getenv("EPOCHS", "1"))  # Default to 1 epoch for testing
    batch_size = int(os.getenv("BATCH_SIZE", "32"))  # Default batch size
    learning_rate = float(os.getenv("LEARNING_RATE", "0.002"))  # Learning rate

    # Initialize hardware manager and determine device
    hardware_manager = HardwareManager()
    device = hardware_manager.get_device()

    # Load the dataset (assuming a function load_dataset exists)
    dataset = load_dataset(project_root)  # Replace with actual dataset loading logic

    # Initialize the trainer
    trainer = GenshinAssistantTrainer(
        dataset=dataset,  # Correct argument
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device
    )
    trainer.model.to(device)  # Move model to appropriate device

    # Check CUDA availability
    print("CUDA Available:", torch.cuda.is_available())

    # Create progress bar for batches
    pbar = tqdm(total=epochs, desc="Training Progress")

    # Train the model
    print(
        f"\nStarting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}"
    )

    for epoch in range(epochs):
        for batch_idx, (queries, responses) in enumerate(trainer.dataloader):  # Unpack correct number of values
            queries = queries.to(device).long()  # Convert to LongTensor
            responses = responses.to(device).long()  # Convert to LongTensor

            # Ensure indices are within the valid range for the embedding layer
            queries = torch.clamp(queries, min=0, max=trainer.model.embedding.num_embeddings - 1)
            responses = torch.clamp(responses, min=0, max=trainer.model.embedding.num_embeddings - 1)

            # Forward pass
            outputs = trainer.model(queries, responses)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, trainer.vocab_size)
            if responses.dim() > 1:
                targets = responses[:, 1:].contiguous().view(-1)  # Skip start token
            else:
                targets = responses.contiguous().view(-1)  # Handle 1D case

            # Ensure outputs and targets have matching batch sizes
            if outputs.size(0) != targets.size(0):
                targets = targets[:outputs.size(0)]

            # Calculate loss
            loss = trainer.criterion(outputs, targets)

            # Check for NaN or inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected:", loss)
                exit()

            # Backward pass
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), max_norm=1.0
            )  # Gradient clipping
            trainer.optimizer.step()

        # Update progress bar
        pbar.update(1)

    # Close the progress bar after training
    pbar.close()

    # Test the model with simple queries
    print("\nTesting the trained model with simple queries:")
    test_queries = [
        "Tell me about Hu Tao",  # Simple character query
        "What is a Dull Blade?",  # Simple weapon query
        "Compare Amber and Lisa",  # Simple comparison
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        response = trainer.generate_response(query)
        print(f"A: {response}")


if __name__ == "__main__":
    main()
