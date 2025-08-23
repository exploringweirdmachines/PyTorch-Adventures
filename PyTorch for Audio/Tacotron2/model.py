import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass

@dataclass
class Tacotron2Config:

    ### Mel Input Features ###
    num_mels: int = 80 

    ### Character Embeddings ###
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    ### Encoder config ###
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5
    
    ### Decoder Config ###
    decoder_rnn_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_dropout_p: float = 0.5

    ### Attention Config ###
    attention_rnn_embed_dim: int = 1024
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1

class LinearNorm(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 w_init_gain="linear"):
        
        super(LinearNorm, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear.weight, 
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear(x)
    
class ConvNorm(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size=1, 
                 stride=1, 
                 padding=None, 
                 dilation=1, 
                 bias=True, 
                 w_init_gain="linear"):
        
        super(ConvNorm, self).__init__()

        if padding is None:
            
            assert (kernel_size % 2 == 1)

            padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            bias=bias
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, config):
        
        super(Encoder, self).__init__()
        
        self.config = config

        self.embeddings = nn.Embedding(config.num_chars, 
                                       config.encoder_embed_dim, 
                                       padding_idx=config.pad_token_id)

        self.convolutions = nn.ModuleList()

        for _ in range(config.encoder_n_convolutions):

            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=config.encoder_embed_dim,
                        out_channels=config.encoder_embed_dim, 
                        kernel_size=config.encoder_kernel_size,
                        stride=1, 
                        padding="same", 
                        dilation=1, 
                        w_init_gain="relu"
                    ),

                    nn.BatchNorm1d(config.encoder_embed_dim), 
                    nn.ReLU(), 
                    nn.Dropout(config.encoder_dropout_p)
                )
            ) 

        self.lstm = nn.LSTM(input_size=config.encoder_embed_dim, 
                            hidden_size=config.encoder_embed_dim//2,
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
    def forward(self, x, input_lengths=None):

        ### Embed Tokens and transpose to (B x E x T) ###
        x = self.embeddings(x).transpose(1,2)

        batch_size, channels, seq_len = x.shape

        if input_lengths is None:
            input_lengths = torch.full((batch_size, ), fill_value=seq_len, device="cpu")

        for block in self.convolutions:
            x = block(x)

        ### Convert to BxLxE ###
        x = x.transpose(1,2)

        ### Pack Padded Sequence so LSTM doesnt Process Pad Tokens ###
        ### This requires data to be sorted in longest to shortest!! ###
        x = pack_padded_sequence(x, input_lengths.to("cpu"), batch_first=True)
    
        ### Pass Data through LSTM ###
        outputs, _ = self.lstm(x)

        ### Pad Packed Sequence ###
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        return outputs
    
class Prenet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 prenet_dim, 
                 prenet_depth,
                 dropout_p=0.5):
        
        super(Prenet, self).__init__()

        self.dropout_p = dropout_p

        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)]

        self.layers = nn.ModuleList()
        
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(

                nn.Sequential(
                    LinearNorm(
                        in_features=in_dim, 
                        out_features=out_dim,
                        bias=False, 
                        w_init_gain="relu"
                    ),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.layers:

            ### Even during inference we leave this dropout enabled to "introduce output variation" ###
            x = F.dropout(layer(x), p=self.dropout_p, training=True)

        return x

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()

        self.conv = ConvNorm(
            in_channels=2, 
            out_channels=attention_n_filters, 
            kernel_size=attention_kernel_size,
            padding="same",
            bias=False, 
            stride=1, 
            dilation=1
        )

        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights_cat):
        attention_weights_cat = self.conv(attention_weights_cat).transpose(1,2)
        attention_weights_cat = self.proj(attention_weights_cat)
        return attention_weights_cat
    
class LocalSensitiveAttention(nn.Module):
    def __init__(self, 
                 attention_dim, 
                 attention_rnn_embed_dim,
                 encoder_embed_dim, 
                 att_location_n_filters,
                 att_location_kernel_size):
        super(LocalSensitiveAttention, self).__init__()

        self.in_proj = nn.Linear(attention_rnn_embed_dim, attention_dim, bias=True)
        self.enc_proj = nn.Linear(encoder_embed_dim, attention_dim, bias=False)
        self.energy_proj = nn.Linear(attention_dim, 1, bias=False)

        self.what_have_i_said = LocationLayer(
            attention_n_filters=att_location_n_filters,
            attention_kernel_size=att_location_kernel_size,
            attention_dim=attention_dim
        )
            
        self.mask_val = -float("inf")
        
        self.reset()

    def reset(self):
        self.enc_proj_cache = None

    def _calculate_alignment_energies(self, 
                                      mel_input, 
                                      encoder_output, 
                                      attention_weights_cat):
        
        ### Query is our Input Mel from previous timestep ###
        mel_proj = self.in_proj(mel_input.unsqueeze(1))

        ### Values is our input character embeddings ###
        ### We can store in cache for each forward pass as this doesnt change ###
        ### for every mel generated timestep ###
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)

        ### Look at the attention weights of the last step and the cumulative ###
        ### attention weights upto the last step to see where I should next ###
        ### place my focus!!! This is what helps the model be monotonic ###
        attention_weights = self.what_have_i_said(attention_weights_cat)
       
        ### Total energy is an accumulation of our importance towards the current ###
        #### mel step broadcasted over all our timesteps of the character embeddings ###
        ### and our location features ### 
        energies = self.energy_proj(
            torch.tanh(mel_proj + self.enc_proj_cache + attention_weights)
        )

        energies = energies.squeeze(-1)
        
        return energies
    
    def forward(self, 
                mel_input,
                encoder_output, 
                attention_weights_cat, 
                mask=None):

        ### Compute Raw Alignment Energies ###
        energies = self._calculate_alignment_energies(mel_input, encoder_output, attention_weights_cat)
 
        if mask is not None:
            energies.masked_fill_(mask.bool(), self.mask_val)

        ### Softmax to create probability Vector ###
        attention_weights = F.softmax(energies, dim=1)

        ### Weight our character embeddings by our probabilities ####
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights
    
class PostNet(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 postnet_num_convs=5, 
                 postnet_filter_size=512, 
                 postnet_kernel_size=5):
        
        super(PostNet, self).__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                ConvNorm(feature_dim, 
                          postnet_filter_size, 
                          kernel_size=postnet_kernel_size, 
                          padding="same",
                          dilation=1, 
                          w_init_gain="tanh"), 

                nn.BatchNorm1d(postnet_filter_size),
                nn.Tanh(), 
                nn.Dropout(0.5)
            )
        )

        for _ in range(postnet_num_convs - 2):
            
            self.convs.append(
                nn.Sequential(
                    ConvNorm(postnet_filter_size, 
                             postnet_filter_size, 
                             kernel_size=postnet_kernel_size, 
                             padding="same",
                             dilation=1, 
                             w_init_gain="tanh"), 

                    nn.BatchNorm1d(postnet_filter_size),
                    nn.Tanh(), 
                    nn.Dropout(0.5)
                )
            )

        self.convs.append(
            nn.Sequential(
                    ConvNorm(postnet_filter_size, 
                             feature_dim, 
                             kernel_size=postnet_kernel_size, 
                             padding="same",
                             dilation=1), 

                    nn.BatchNorm1d(feature_dim),
                    nn.Dropout(0.5)
                )
        )
    
    def forward(self, x):
        
        ### Transpose from (B x T x C) to (B x C x T) ###
        x = x.transpose(1,2)
        for conv_block in self.convs:
            x = conv_block(x)
        x = x.transpose(1,2)
        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        ### Predictions from previous timestep passed through a few linear layers ###
        self.prenet = Prenet(input_dim=config.num_mels,
                             prenet_dim=config.decoder_prenet_dim, 
                             prenet_depth=config.decoder_prenet_depth,
                             dropout_p=config.decoder_prenet_dropout_p)

        self.attention_rnn = nn.LSTMCell(
            input_size=config.decoder_prenet_dim + config.encoder_embed_dim,
            hidden_size=config.attention_rnn_embed_dim
        )

        self.attention_dropout = nn.Dropout(p=config.attention_dropout_p)

        ### Local Sensitive Attention Module ###
        self.attention = LocalSensitiveAttention(attention_dim=config.attention_dim,
                                                 attention_rnn_embed_dim=config.attention_rnn_embed_dim,
                                                 encoder_embed_dim=config.encoder_embed_dim, 
                                                 att_location_n_filters=config.attention_location_n_filters,
                                                 att_location_kernel_size=config.attention_location_kernel_size)
        
        self.decoder_rnn = nn.LSTMCell(
            input_size=config.attention_rnn_embed_dim + config.encoder_embed_dim,
            hidden_size=config.decoder_rnn_embed_dim
        )
        
        self.decoder_dropout = nn.Dropout(p=config.decoder_dropout_p)

        self.mel_proj = LinearNorm(
            in_features=config.decoder_rnn_embed_dim + config.encoder_embed_dim, 
            out_features=config.num_mels
        )

        self.stop_pred = LinearNorm(
            in_features=config.decoder_rnn_embed_dim + config.encoder_embed_dim, 
            out_features=1, 
            w_init_gain="sigmoid"
        )

    def _bos_frame(self, B):

        ### Start predicting from zero vector ###
        start_frame_zeros = torch.zeros(B, 1, self.config.num_mels)

        return start_frame_zeros
    
    def _init_decoder(self, encoder_outputs, encoder_mask=None):
 
        batch_size, seq_len, _ = encoder_outputs.shape
        device = encoder_outputs.device

        self.attention_hidden = torch.zeros(batch_size, self.config.attention_rnn_embed_dim, device=device)
        self.attention_cell = torch.zeros(batch_size, self.config.attention_rnn_embed_dim, device=device)

        self.decoder_hidden = torch.zeros(batch_size, self.config.decoder_rnn_embed_dim, device=device)
        self.decoder_cell = torch.zeros(batch_size, self.config.decoder_rnn_embed_dim, device=device)

        self.attention_weights = torch.zeros(batch_size, seq_len, device=device)
        self.cumulative_attn_weights = torch.zeros(batch_size, seq_len, device=device)
        self.attention_context = torch.zeros(batch_size, self.config.encoder_embed_dim, device=device)

        self.encoder_outputs = encoder_outputs
        self.mask = encoder_mask

    def _step(self, decoder_input):

        decoder_input = torch.cat([decoder_input, self.attention_context], dim=-1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            decoder_input, (self.attention_hidden, self.attention_cell)
        )
        
        self.attention_hidden = self.attention_dropout(self.attention_hidden)

        attention_weights_cat = torch.cat([self.attention_weights.unsqueeze(1), 
                                           self.cumulative_attn_weights.unsqueeze(1)], dim=1)
        
        self.attention_context, self.attention_weights = self.attention(
            self.attention_hidden, self.encoder_outputs, attention_weights_cat, self.mask
        )

        self.cumulative_attn_weights = self.cumulative_attn_weights + self.attention_weights

        decoder_input = torch.cat([self.attention_hidden, self.attention_context], dim=-1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )

        self.decoder_hidden = self.decoder_dropout(self.decoder_hidden)

        decoder_hidden_attn_context = torch.cat(
            [self.decoder_hidden, self.attention_context], dim=1
        )

        mel_pred = self.mel_proj(decoder_hidden_attn_context)
        stop_pred = self.stop_pred(decoder_hidden_attn_context)

        return mel_pred, stop_pred, self.attention_weights

    def forward(self,
                encoder_outputs,
                encoder_mask, 
                mels):
        
        ### When Decoding Start with Zero Feature Vector ###
        start_feature_vector = self._bos_frame(mels.shape[0]).to(mels.device)
        mels_w_start = torch.cat([start_feature_vector, mels], dim=1)
        
        self._init_decoder(encoder_outputs, encoder_mask)

        ### Create lists to store Intermediate Outputs ###
        mel_outs, stop_outs, attention_weights = [], [], []
        
        ### Teacher forcing for T Steps ###
        T_dec = mels.shape[1]

        ### Project Mel Spectrograms by PreNet ###
        mel_proj = self.prenet(mels_w_start)

        ### Loop through T timesteps ###
        for t in range(T_dec):
            
            ### Every forward pass reset our attention encoder embeddings ###
            if t == 0:
                self.attention.reset()

            ### Always grab our true mel as our input (teacher forcing) ###
            ### rather than use previous predicted one ###
            step_input = mel_proj[:, t, :]

            ### Compute our mel output, our stop token prob and attention weight ###
            mel_out, stop_out, attention_weight = self._step(step_input)

            ### Store in buffer ###
            mel_outs.append(mel_out)
            stop_outs.append(stop_out)
            attention_weights.append(attention_weight)

        #### Stack all timesteps together ###
        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)

        return mel_outs, stop_outs, attention_weights
    

    @torch.inference_mode()
    def inference(self, encoder_outputs, max_steps=1000):
        B, _, _ = encoder_outputs.shape

        start_feature_vector = self._decoder_start_frame(B).to(encoder_outputs.device)

        self._init_decoder(encoder_outputs)
        
        mel_outs, stop_outs, attention_weights = [], [], []
        
        # The PreNet is applied outside the loop on the start vector.
        # This is because the step function expects the projected input.
        mel_input = self.prenet(start_feature_vector).squeeze(1) 
        self.attention.reset()

        while True:
            # We pass the pre-processed input to _step
            mel_out, stop_out, attention_weight = self._step(mel_input)

            mel_outs.append(mel_out)
            stop_outs.append(stop_out)
            attention_weights.append(attention_weight)
            
            # Use the generated mel_out from this step as the input for the next step
            # You must project it with the Prenet before the next iteration
            mel_input = self.prenet(mel_out.unsqueeze(1)).squeeze(1)
            
            if torch.sigmoid(stop_out)[0] > 0.5:
                print(f"Stop token predicted at step {len(mel_outs)}")
                break
            elif len(mel_outs) > max_steps:
                print("Reached Max Steps!")
                break

        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)

        return mel_outs, attention_weights

class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.postnet = PostNet(feature_dim=config.num_mels, 
                               postnet_num_convs=config.decoder_postnet_num_convs, 
                               postnet_filter_size=config.decoder_postnet_n_filters, 
                               postnet_kernel_size=config.decoder_postnet_kernel_size)

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask):

        encoder_padded_outputs = self.encoder(text, input_lengths)
        mel_outs, stop_outs, attention_weights = self.decoder(
            encoder_padded_outputs, encoder_mask, mels
        )   

        mel_residual = self.postnet(mel_outs)
        mel_outs_postnet = mel_outs + mel_residual

        if decoder_mask is not None: 

            decoder_mask_exp = decoder_mask.unsqueeze(-1).expand(decoder_mask.shape[0], decoder_mask.shape[1], self.config.num_mels)
            mel_outs = mel_outs.masked_fill(decoder_mask_exp.bool(), 0.0)
            mel_outs_postnet = mel_outs_postnet.masked_fill(decoder_mask_exp.bool(), 0.0)
            stop_outs = stop_outs.masked_fill(decoder_mask.bool(), 1e3)


        return mel_outs, mel_outs_postnet, stop_outs, attention_weights 
    
    @torch.inference_mode()
    def inference(self, text, max_steps=1000):
        encoder_outputs = self.encoder(text)
        
        mel_out, attention_weights = self.decoder.inference(encoder_outputs=encoder_outputs,
                                                            max_steps=max_steps)
        
        mel_residual = self.postnet(mel_out)

        mel = mel_out + mel_residual

        return mel, attention_weights


if __name__ == "__main__":

    from dataset import TTSDataset, TTSCollator

    dataset = TTSDataset("data/test_metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=TTSCollator())
    # model = Tacotron2(Tacotron2Config())

    # for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in loader:

    #     mel_outs, mel_postnet_outs, stop_tokens, attention_weights = model(
    #         text_padded, input_lengths, mel_padded, encoder_mask, decoder_mask
    #     )
        
    #     print(mel_outs.shape)
    #     print(mel_postnet_outs.shape)
    #     print(stop_tokens.shape)
    #     print(attention_weights.shape)

    #     break

    
    import torch 
    from tokenizer import Tokenizer
    import matplotlib.pyplot as plt
    import torchaudio.transforms as AT
    import torchaudio
    from torch.utils.data import DataLoader

    config = Tacotron2Config()
    model = Tacotron2(config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=TTSCollator()
    )

    
    for text_padded, input_lengths, mel_padded, gate_padded, enc_mask, dec_mask in dataloader:

        mel_outs, post_outs, gate_outs, alignments = model(
            text_padded, input_lengths, mel_padded, enc_mask, dec_mask
        )



        break
        