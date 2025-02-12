import sys
import logging

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from pathlib import Path
from scraper.api_client import GenshinAPIClient
from preprocessing.data_processor import GenshinDataProcessor
from models.genshin_model import GenshinAITrainer

def main():
    try:
        # Setup paths
        project_dir = Path(__file__).parent.parent
        data_dir = project_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Create necessary directories
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        processed_dir = data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

        models_dir = data_dir / "models"
        models_dir.mkdir(exist_ok=True)

        print("1. Fetching data from Genshin Impact APIs...")
        client = GenshinAPIClient(data_dir)
        data = client.fetch_all_data()

        if not data:
            print("Error: No data was fetched from the APIs")
            sys.exit(1)

        print("\n2. Processing data for AI training...")
        processor = GenshinDataProcessor(data_dir)
        processed_data = processor.load_and_process_data()

        if not processed_data:
            print("Error: No data was processed successfully")
            sys.exit(1)

        print("\nProcessed datasets:")
        for name, df in processed_data.items():
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            print(f"- {name}:")
            print(f"  * {len(df)} records")
            print(f"  * {len(numeric_cols)} numeric features: {list(numeric_cols)}")
            print(f"  * Sample data:\n{df.head(2)}\n")

        print("\n3. Training AI models...")
        trainer = GenshinAITrainer(data_dir)
        try:
            trainer.train_models(epochs=50)  # Reduced epochs for testing
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            sys.exit(1)

        print("\n4. Generating embeddings...")
        for dataset_name in processed_data.keys():
            try:
                embeddings = trainer.generate_embeddings(dataset_name)
                print(f"Generated embeddings for {dataset_name}:")
                print(f"- Shape: {embeddings.shape}")
                if len(embeddings):
                    print(f"- Sample values:\n{embeddings[:2]}\n")
            except Exception as e:
                print(f"Error generating embeddings for {dataset_name}: {e}")

        print("\nProcess completed successfully!")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
