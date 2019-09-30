import torch
from torch import nn
import torchvision
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    """
    Convolutional Autoencoder network.
    """

    def __init__(self, features_dim, num_regions):
        super(AutoEncoder, self).__init__()
        self.linear_transform = weight_norm(nn.Linear(4096, 1024))
        self.encoder = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(15, 26), stride=2)
        # decoder (deconvolution)
        #self.decoder = Conv2DTranspose(1, (15, 26), strides=2, padding='valid')

    def forward(self, x):
        #print('X SHAPE', x.shape)
        x = self.linear_transform(x)
        #print('X LINEAR', x.shape)
        #x = x.unsqueeze(-1)
        x = x[None, None]
        #print('X INPUT', x.shape)
        # x is fed to the autoencoder
        x = self.encoder(x)
        # no decoding is needed, we use 5 topics
        #x = self.decoder(x)
        return x.squeeze(0)


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, topic_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        :param topic_dim: size of the topic vector
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.topic_att = weight_norm(nn.Linear(topic_dim, attention_dim)) # linear layer to transform topic vectors
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden, topic_feats):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :param topic_feats: topics, a tensor of dimension (500,)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 15, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att3 = self.topic_att(topic_feats)
        att = self.full_att(self.dropout(self.relu(att1 + att2 + att3.unsqueeze(1)))).squeeze(2)  # (batch_size, 15)
        alpha = self.softmax(att)  # (batch_size, 15)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=4096, dropout=0.5, num_regions=15, topic_dim=500):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param num_regions: number of regions used to encode images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_regions = num_regions
        self.topic_dim = topic_dim

        self.autoencoder = AutoEncoder(num_regions, features_dim)
        self.attention = Attention(features_dim, decoder_dim, attention_dim, topic_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        #print('SIZE', embed_dim + features_dim + decoder_dim)

        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + decoder_dim, decoder_dim, bias=True) # top down attention LSTMCell

        self.language_model = nn.LSTMCell(features_dim + decoder_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(1, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(1, self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim) 100 x 15 x 4096
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length) 100 x 5 x 52
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1) 100 x 5
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(device)  # (batch_size, num_pixels, encoder_dim) 100 x 4096

        decode_lengths = (caption_lengths - 1).tolist()

        #print('DECODE LENGTHS', decode_lengths)

        # make sure that indexing here works properly!!!! right images are combined with right paragraphs!

#        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
#        image_features = image_features[sort_ind]
#        image_features_mean = image_features_mean[sort_ind]
#        encoded_captions = encoded_captions[sort_ind]

        predictions = torch.zeros(batch_size, 5, vocab_size).to(device)
        predictions1 = torch.zeros(batch_size, 5, vocab_size).to(device)

        for num, each_sample in enumerate(decode_lengths):

            h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
            h2, c2 = self.init_hidden_state(batch_size)

            this_image_features = image_features[num]
            this_image_features_mean = image_features_mean[num].unsqueeze(0)
            this_encoded_captions = encoded_captions[num]
            embeddings = self.embedding(this_encoded_captions)

            #print('F', this_image_features, this_image_features.shape)
            #print('MEAN F', this_image_features_mean, this_image_features_mean.shape)
            #print('THIS ENCODING', this_encoded_captions, this_encoded_captions.shape)
            #print('THIS EMBEDDING', embeddings, embeddings.shape)
            #print('SAMPLE', each_sample)

            topics = self.autoencoder(this_image_features)

            #print('TOPICS', topics)
            #print('TOPICS SHAPE', topics[0].squeeze(0).shape)

            list_predictions = []
            list_predictions1 = []

            for t in range(5):

                this_topic = topics[t]

                last_word_index = each_sample[t]
                last_word_embedding = embeddings[t][last_word_index].unsqueeze(0)

                h1,c1 = self.top_down_attention(
                    torch.cat([h2, this_image_features_mean, last_word_embedding], dim=1),(h1, c1))

                attention_weighted_encoding = self.attention(this_image_features, h1, this_topic)
                print('ATT WEIGHTED ENCODING', attention_weighted_encoding)

                #break

                preds1 = self.fc1(self.dropout(h1))

                h2,c2 = self.language_model(
                    torch.cat([attention_weighted_encoding, h1], dim=1),
                    (h2, c2))
                preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)

                list_predictions.append(preds)
                list_predictions1.append(preds1)

            paragraph_predictions = torch.cat(list_predictions, dim=0)
            paragraph_predictions1 = torch.cat(list_predictions1, dim=0)
            #print(paragraph_predictions, paragraph_predictions.shape)

            predictions[num, :, :] = paragraph_predictions
            predictions1[num, :, :] = paragraph_predictions1

        print(predictions[0:3], predictions.shape)

        return predictions, predictions1, encoded_captions, decode_lengths
