from pathlib import Path
from models.genshin_assistant import GenshinAssistantTrainer
import os
from tqdm import tqdm

def main():
    # Get configuration from environment variables with smaller default values
    epochs = int(os.getenv("EPOCHS", "1"))  # Default to 1 epoch for testing
    batch_size = int(os.getenv("BATCH_SIZE", "32"))  # Larger batch size for faster training
    learning_rate = float(os.getenv("LEARNING_RATE", "0.002"))  # Slightly higher learning rate
    
    # Initialize trainer
    data_dir = Path(__file__).parent.parent
    trainer = GenshinAssistantTrainer(
        data_dir,
        embedding_dim=128,  # Reduced from 256
        hidden_dim=256      # Reduced from 512
    )
    
    # Train the model
    print(f"\nStarting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
    trainer.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Test the model with shorter queries first
    print("\nTesting the trained model with simple queries:")
    test_queries = [
        "Tell me about Hu Tao",  # Simple character query
        "What is a Dull Blade?", # Simple weapon query
        "Compare Amber and Lisa", # Simple comparison
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        response = trainer.generate_response(query)
        print(f"A: {response}")

if __name__ == "__main__":
    main()