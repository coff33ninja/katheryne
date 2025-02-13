from pathlib import Path
import os
import torch
import sys
from models.decoders import HardwareManager

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.genshin_assistant import GenshinAssistantTrainer
from tqdm import tqdm


def main():
    # Get configuration from environment variables with sensible defaults
    epochs = int(os.getenv("EPOCHS", "1"))  # Default to 1 epoch for testing
    batch_size = int(os.getenv("BATCH_SIZE", "32"))  # Default batch size
    learning_rate = float(os.getenv("LEARNING_RATE", "0.002"))  # Learning rate

    # Initialize hardware manager and determine device
    hardware_manager = HardwareManager()
    device = hardware_manager.get_device()

    # Initialize the trainer
    trainer = GenshinAssistantTrainer(
        data_dir=project_root, embedding_dim=128, hidden_dim=256
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
        for batch_idx, (queries, responses, _) in enumerate(trainer.dataloader):
            queries = queries.to(device)
            responses = responses.to(device)

            # Forward pass
            outputs = trainer.model(queries, responses)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, trainer.dataset.vocab_size)
            targets = responses[:, 1:].contiguous().view(-1)  # Skip start token

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
