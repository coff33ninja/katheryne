import argparse
import json
import torch
from pathlib import Path
from models.genshin_assistant import GenshinAssistant
from dataset import GenshinAssistantDataset

def load_model(model_path: Path, vocab_path: Path):
    """Load the trained model and vocabulary"""
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Initialize model
    model = GenshinAssistant(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        failover=True  # Use light model for inference
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

def generate_response(model, vocab, query: str, max_length: int = 100):
    """Generate a response for the given query"""
    # Convert query to tensor
    idx_to_word = {v: k for k, v in vocab.items()}
    words = ['<BOS>'] + query.lower().split() + ['<EOS>']
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    query_tensor = torch.tensor(indices).unsqueeze(0)
    
    # Generate response
    with torch.no_grad():
        response_indices = []
        current_token = torch.tensor([[vocab['<BOS>']]])
        
        encoder_output, hidden = model.encoder(model.embedding(query_tensor))
        decoder_hidden = hidden
        
        for _ in range(max_length):
            output, decoder_hidden = model.decoder(
                model.embedding(current_token),
                decoder_hidden,
                encoder_output
            )
            
            # Get next token
            current_token = output.argmax(2)
            token_idx = current_token.item()
            
            if token_idx == vocab['<EOS>']:
                break
                
            response_indices.append(token_idx)
        
        # Convert indices back to words
        response_words = [idx_to_word[idx] for idx in response_indices]
        response_text = ' '.join(response_words)
        
        # Try to parse as JSON
        try:
            response_data = json.loads(response_text)
            return response_data
        except json.JSONDecodeError:
            return {"text": response_text}

def main():
    parser = argparse.ArgumentParser(description='Test the Genshin Assistant model')
    parser.add_argument('--query', type=str, required=True, help='Query to test')
    parser.add_argument('--model-path', type=str, 
                       default='models/assistant_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab-path', type=str,
                       default='models/assistant_vocab.json',
                       help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    # Load model and vocabulary
    model_path = Path(args.model_path)
    vocab_path = Path(args.vocab_path)
    
    if not model_path.exists():
        print(json.dumps({"error": "Model file not found"}))
        return
    
    if not vocab_path.exists():
        print(json.dumps({"error": "Vocabulary file not found"}))
        return
    
    try:
        model, vocab = load_model(model_path, vocab_path)
        response = generate_response(model, vocab, args.query)
        print(json.dumps(response))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()