import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = nn.RNN(input_size = embed_size, 
                          hidden_size = hidden_size, 
                          num_layers = rnn_layers, 
                          batch_first=True) if rnn_type == nn.RNN else nn.LSTM(input_size = embed_size,
                                                                                      hidden_size = hidden_size, 
                                                                                       num_layers = rnn_layers,
                                                                                       batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
    
    
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """       
          
          
        '''
        B - batch size
        L - sequence length
        E - embedding dim
        H - hidden dim
        V - vocab size
        '''
        # indices: (B, L)
        embeds = self.embedding(indices)
        # embeds: (B, L, E) in padded form
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed_embeds)
        # output: (B, L, H), hidden: (B, H) in packed form
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)
        # logits: (B, L, V)
        return logits
        

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """

        self.eval()

        # encode prefix
        if len(prefix) == 0:
            indices = [self.dataset.bos_id]
        else:
            indices = self.dataset.text2ids(prefix)
        indices = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        
        # generate hidden for prefix
        embeds = self.embedding(indices)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output[:, -1])
        # sample new token from logits
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        new_indices = torch.multinomial(probs, 1)
        indices = torch.cat([indices, new_indices], dim=1)
        
        # 2 stopping conditions: reaching max len or getting <eos> token
        while indices.shape[1] < self.max_length:
            if new_indices.item() == self.dataset.eos_id:
                break

            # process newly obtained token
            embeds = self.embedding(new_indices)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output[:, -1])
            # sample the next token from logits
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            new_indices = torch.multinomial(probs, 1)
            indices = torch.cat([indices, new_indices], dim=1)
        
        # decode result to a string
        generated = self.dataset.ids2text(indices.squeeze())
               
        return generated
