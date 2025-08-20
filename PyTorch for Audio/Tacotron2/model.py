import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass

@dataclass
class Tacotron2Config:

    num_chars: int = 67
    num_mels: int = 80 

    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5
    
    decoder_embed_dim: int = 512
    decoder_hidden_size: int = 1024
    
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5

    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5

    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

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

        self.embeddings = nn.Embedding(config.num_chars, config.encoder_embed_dim)

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
            input_lengths = torch.full((batch_size, ), fill_value=seq_len, device=x.device)

        for block in self.convolutions:
            x = block(x)

        ### Convert to BxLxE ###
        x = x.transpose(1,2)

        ### Pack Padded Sequence so LSTM doesnt Process Pad Tokens ###
        ### This requires data to be sorted in longest to shortest!! ###
        x = pack_padded_sequence(x, input_lengths, batch_first=True)
    
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

class LocalSensitiveAttention(nn.Module):
    def __init__(self, 
                 attention_dim, 
                 decoder_hidden_size,
                 encoder_hidden_size, 
                 location_feature_dim):
        super(LocalSensitiveAttention, self).__init__()

        self.Q = nn.Linear(decoder_hidden_size, attention_dim, bias=True)
        self.V = nn.Linear(encoder_hidden_size, attention_dim, bias=False)

        self.loc_feat = nn.Conv1d(in_channels=1, out_channels=location_feature_dim, 
                           kernel_size=31, stride=1, padding="same", bias=False)
        
        self.loc_feat_proj = nn.Linear(location_feature_dim, attention_dim, bias=False)
        
        
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.reset()

    def reset(self):
        self.values_cache = None

    def _calculate_alignment_energies(self, query, values, cumulative_attention_weights, mask=None):
        
        ### Query is our Input Mel from previous timestep ###
        query = self.Q(query.unsqueeze(1))

        ### Values is our input character embeddings ###
        ### We can store in cache for each forward pass as this doesnt change ###
        ### for every mel generated timestep ###
        if self.values_cache is None:
            self.values_cache = self.V(values)

        ### Use convolution and projeciton to look at our accumulation of how much attention has ###
        ### been placed already on character embeddings. This incentivizes mononotic ###
        ### progression in our attention computation ###
        location_features = self.loc_feat(cumulative_attention_weights.unsqueeze(1))
        
        ### Project the location features to attention dim ###
        location_features = self.loc_feat_proj(location_features.transpose(1,2))

        ### Total energy is an accumulation of our importance towards the current ###
        #### mel step (query) broadcasted over all our timesteps of the character embeddings ###
        ### and our location features ### 
        energies = self.v(torch.tanh(query + self.values_cache + location_features)).squeeze(-1)

        ### If we had padding in our character embeddings, we mask them out here ###
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        
        return energies
    
    def forward(self, query, values, cumulative_attention_weights, mask=None):

        ### Compute Raw Alignment Energies ###
        energies = self._calculate_alignment_energies(query, values, cumulative_attention_weights, mask)
        
        ### Softmax to create probability Vector ###
        attention_weights = F.softmax(energies, dim=1)

        ### Weight our character embeddings by our probabilities ####
        attention_context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return attention_context, attention_weights
    
class PostNet(nn.Module):
    def __init__(self, feature_dim, postnet_num_convs=5, postnet_filter_size=512, postnet_kernel_size=5):
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
        self.prenet = Prenet(input_dim=self.config.num_mels,
                             prenet_dim=self.config.decoder_prenet_dim, 
                             prenet_depth=self.config.decoder_prenet_depth)

        ### LSTMs Module to Process Concatenated PreNet output and Attention Context Vector ###
        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(config.decoder_prenet_dim + config.decoder_embed_dim, config.decoder_hidden_size), 
                nn.LSTMCell(config.decoder_hidden_size, config.decoder_hidden_size)
            ]
        )
   
        ### Local Sensitive Attention Module ###
        self.attention = LocalSensitiveAttention(attention_dim=config.attention_dim,
                                                 decoder_hidden_size=config.decoder_hidden_size, 
                                                 encoder_hidden_size=config.encoder_embed_dim, 
                                                 location_feature_dim=config.attention_location_n_filters)
        
        ### Predict Next Mel ###
        self.feature_linear = LinearNorm(config.decoder_hidden_size + config.encoder_embed_dim, config.num_mels)
        self.stop_linear = LinearNorm(config.decoder_hidden_size + config.encoder_embed_dim, 1, w_init_gain="sigmoid")

        ### Post Process Predicted Mel ###
        self.postnet = PostNet(feature_dim=config.num_mels, 
                               postnet_num_convs=config.decoder_postnet_num_convs,
                               postnet_filter_size=config.decoder_postnet_n_filters,
                               postnet_kernel_size=config.decoder_postnet_kernel_size)

    def _init_decoder(self, encoder_outputs):

        B, S, E = encoder_outputs.shape

        ### Initialize Memory for two LSTM Cells ###
        self.h = [torch.zeros(B, self.config.decoder_hidden_size) for _ in range(2)]
        self.c = [torch.zeros(B, self.config.decoder_hidden_size) for _ in range(2)]

        ### Initialize Cumulative Attention ###
        self.cumulative_attn_weight = torch.zeros(B,S)
        self.attn_context = torch.zeros(B, self.config.encoder_embed_dim)

        ### Store Encoder Outputs ##
        self.encoder_outputs = encoder_outputs

    def _decoder_start_frame(self, B):

        ### Start predicting from zero vector ###
        start_frame_zeros = torch.zeros(B, 1, self.config.num_mels)

        return start_frame_zeros

    def _step(self, mel_step):

        ### For each step, concatenate onto our current mel step our weighted ###
        ### character embedding context from the previous step (zero initially) ###
        ### This provides our RNN both the input to generate the next mel and also ###
        ### the previously weighted encoder information ###
        rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)

        ### Pass through manually throuhgh our two LSTM layers ###
        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))
        self.h[1], self.c[1] = self.rnn[1](self.h[0], (self.h[1], self.c[1]))
        rnn_output = self.h[1]

        ### With the new output from our mel + its history, compute attention ###
        ### to see how the new step is related to the character embeddings ###
        attention_context, attention_weights = self.attention(
            rnn_output, 
            self.encoder_outputs, 
            self.cumulative_attn_weight, 
            mask=self.encoder_mask
        )       

        ### Store the attention context for the next step ###
        self.attn_context = attention_context

        ### Cumulatively add our attention weight for the current step to all previous ###
        ### steps so we can see how much emphasis was placed on each character embedding ###
        self.cumulative_attn_weight = self.cumulative_attn_weight + attention_weights

        ### Concatenate onto our final rnn output our newly compute attention context ###
        ### so we can use it to predict our next mel output as well as stop tokens ###
        linear_input = torch.cat((rnn_output, self.attn_context), dim=1)
        mel_out = self.feature_linear(linear_input)
        stop_out = self.stop_linear(linear_input)

        return mel_out, stop_out, attention_weights


    def forward(self,
                encoder_outputs,
                encoder_mask, 
                mels, 
                decoder_mask):
        
        ### When Decoding Start with Zero Feature Vector ###
        start_feature_vector = self._decoder_start_frame(mels.shape[0])
        mels_w_start = torch.cat([start_feature_vector, mels], dim=1)
        
        self._init_decoder(encoder_outputs)
        self.encoder_mask = encoder_mask

        ### Create lists to store Intermediate Outputs ###
        mel_outs, stop_tokens, attention_weights = [], [], []
        
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
            stop_tokens.append(stop_out)
            attention_weights.append(attention_weight)

        #### Stack all timesteps together ###
        mel_outs = torch.stack(mel_outs, dim=1)
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)

        ### Further refine our output mel with the postnet ###
        mel_residual = self.postnet(mel_outs)

        ### Mask out for all the timesteps in our mel inputs that were pad###
        decoder_mask = decoder_mask.unsqueeze(-1).bool()
        mel_out = mel_outs.masked_fill(decoder_mask, 0.0)
        mel_residual = mel_residual.masked_fill(decoder_mask, 0.0)
        attention_weights = attention_weights.masked_fill(decoder_mask, 0.0)
        stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3) # Large value because pad means its over so predict stop

        return mel_outs, mel_residual, stop_tokens, attention_weights
    
class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask):

        encoder_padded_outputs = self.encoder(text, input_lengths)
        mel_outs, mel_residual, stop_tokens, attention_weights = self.decoder(
            encoder_padded_outputs, encoder_mask, mels, decoder_mask
        )

        mel_postnet_outs = mel_outs + mel_residual

        return mel_outs, mel_postnet_outs, stop_tokens, attention_weights 

if __name__ == "__main__":

    from dataset import TTSDataset, TTSCollator

    dataset = TTSDataset("data/test_metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=TTSCollator())
    model = Tacotron2(Tacotron2Config())

    for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in loader:

        mel_outs, mel_postnet_outs, stop_tokens, attention_weights = model(
            text_padded, input_lengths, mel_padded, encoder_mask, decoder_mask
        )
        
        print(mel_outs.shape)
        print(mel_postnet_outs.shape)
        print(stop_tokens.shape)
        print(attention_weights.shape)

        break