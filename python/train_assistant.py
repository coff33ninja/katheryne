import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from .models.genshin_assistant import GenshinAssistant
from .dataset import GenshinAssistantDataset

def main():
    # Auto-detect if we should use failover mode
    if torch.cuda.is_available():
        device = torch.device("cuda")
        failover = False
    else:
        device = torch.device("cpu")
        failover = True

    print(f"Using device: {device}")
    print(f"Using {'light' if failover else 'heavy'} model configuration")

    data_dir = Path(__file__).parent.parent

    trainer = GenshinAssistantTrainer(
        data_dir=data_dir,
        embedding_dim=128,
        hidden_dim=256,
        failover=failover
    )

    trainer.train(epochs=1, batch_size=32, learning_rate=0.002)

class GenshinAssistantTrainer:
    def __init__(self, data_dir: Path, embedding_dim: int = 128, hidden_dim: int = 256, failover: bool = None):
        self.data_dir = Path(data_dir)

        # Auto-select failover mode if not specified
        if failover is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                failover = False
            else:
                self.device = torch.device("cpu")
                failover = True
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device} with {'light' if failover else 'heavy'} configuration")

        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.dataset = GenshinAssistantDataset(self.data_dir)
        self.model = GenshinAssistant(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            failover=failover
        ).to(self.device)

        # Save the vocabulary
        vocab_path = self.models_dir / "assistant_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset.vocab, f, ensure_ascii=False, indent=2)

    def train(self, epochs: int = 1, batch_size: int = 32, learning_rate: float = 0.002):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.vocab["<PAD>"])

        best_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (queries, responses, _) in enumerate(pbar):
                queries = queries.to(self.device)
                responses = responses.to(self.device)

                outputs = self.model(queries, responses)
                outputs = outputs.view(-1, self.dataset.vocab_size)
                targets = responses[:, 1:].contiguous().view(-1)  # skip start token

                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint(epoch, avg_loss, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, avg_loss)

    def generate_response(self, query: str, max_length: int = 64) -> str:
        self.model.eval()
        query_tensor = self.dataset._text_to_tensor(query).unsqueeze(0).to(self.device)
        current_token = torch.LongTensor([[self.dataset.vocab["<START>"]]]).to(self.device)
        response = [self.dataset.vocab["<START>"]]

        with torch.no_grad():
            src_embedded = self.model.embedding_dropout(self.model.embedding(query_tensor))
            encoder_outputs, (hidden, cell) = self.model.encoder(src_embedded)

            if not self.model.failover:
                batch_size = query_tensor.size(0)
                hidden = hidden.view(2, 2, batch_size, -1)\
                            .transpose(1, 2).contiguous().view(2, batch_size, -1)
                cell = cell.view(2, 2, batch_size, -1)\
                           .transpose(1, 2).contiguous().view(2, batch_size, -1)
                decoder_hidden = hidden
                decoder_cell = cell
            else:
                decoder_hidden = hidden
                decoder_cell = cell
                encoder_final_hidden = hidden[-1]

            for _ in range(max_length):
                token_embedded = self.model.embedding_dropout(self.model.embedding(current_token))

                if not self.model.failover:
                    attn_weights = self.model.decoder.compute_attention(decoder_hidden[-1], encoder_outputs)
                    context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
                    decoder_input = torch.cat([token_embedded, context], dim=2)
                    output, (decoder_hidden, decoder_cell) = self.model.decoder(
                        decoder_input,
                        (decoder_hidden, decoder_cell),
                        encoder_outputs
                    )
                else:
                    output, (decoder_hidden, decoder_cell) = self.model.decoder(
                        token_embedded,
                        (decoder_hidden, decoder_cell),
                        encoder_final_hidden
                    )

                next_token = output.argmax(dim=2)
                current_token = next_token
                token_idx = next_token.item()
                response.append(token_idx)

                if token_idx == self.dataset.vocab["<END>"]:
                    break

        idx_to_word = {v: k for k, v in self.dataset.vocab.items()}
        response_words = [idx_to_word[idx] for idx in response[1:-1]]  # skip start and end tokens
        return " ".join(response_words)

    def _collate_fn(self, batch):
        queries, responses, types = zip(*batch)
        return torch.stack(queries), torch.stack(responses), types

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss
        }
        checkpoint_path = self.models_dir / (
            "assistant_best.pt" if is_best else f"assistant_checkpoint_{epoch+1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()
