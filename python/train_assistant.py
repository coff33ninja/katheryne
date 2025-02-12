from pathlib import Path
from models.genshin_assistant import GenshinAssistantTrainer
import os
from tqdm import tqdm

def main():
    # Get configuration from environment variables
    epochs = int(os.getenv("EPOCHS", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
    
    # Initialize trainer
    data_dir = Path(__file__).parent.parent
    trainer = GenshinAssistantTrainer(data_dir)
    
    # Train the model
    print(f"\nStarting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
    trainer.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Test the model
    print("\nTesting the trained model:")
    test_queries = [
        "Tell me about Hu Tao's abilities",
        "What's the best weapon for Raiden Shogun?",
        "Compare Ganyu and Ayaka",
        "Suggest a team comp for Abyss floor 12"
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        response = trainer.generate_response(query)
        print(f"A: {response}")

if __name__ == "__main__":
    main()