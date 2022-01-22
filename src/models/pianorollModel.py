from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend

import os
import tensorflow as tf

class PianoRollModel():

    def __init__(
        self,
        model_latent_dim,
        hidden_layers             =  1,
        hidden_layers_dim         = 32,
        
        model_input_dim           = 44,

        encoder_use_batch_norm    = True,
        encoder_use_dropout       = True,
        encoder_dropout_rate      = 0.2,

        decoder_use_batch_norm    = True,
        decoder_use_dropout       = False,
        decoder_dropout_rate      = 0.2,

        model_output_dim          = 44
        ):

        # Save init arguments as self variables
        self.model_hidden_layers       = hidden_layers
        self.model_hidden_layers_dim   = hidden_layers_dim


        self.model_input__dim          = model_input_dim

        self.encoder_input_dim         = model_input_dim
        self.encoder_hidden_layers     = hidden_layers
        self.encoder_hidden_layers_dim = hidden_layers_dim
        self.encoder_output_dim        = model_latent_dim
        self.encoder_use_batch_norm    = encoder_use_batch_norm
        self.encoder_use_dropout       = encoder_use_dropout
        self.encoder_dropout_rate      = encoder_dropout_rate

        self.model_latent_dim          = model_latent_dim

        self.decoder_input_dim         = model_latent_dim
        self.decoder_hidden_layers     = hidden_layers
        self.decoder_hidden_layers_dim = hidden_layers_dim
        self.decoder_output_dim        = model_output_dim
        self.decoder_use_batch_norm    = decoder_use_batch_norm
        self.decoder_use_dropout       = decoder_use_dropout
        self.decoder_dropout_rate      = decoder_dropout_rate

        self.model_output_dim          = model_output_dim

        # Save and create important paths 
        self.model_path = os.path.dirname(os.path.abspath(__file__))
        self.model_imgs_path = '{}/imgs_pianoroll/latent_{}/hidden_{}_dim_{}'.format(self.model_path, self.model_latent_dim, self.model_hidden_layers, self.model_hidden_layers_dim)
        if os.path.exists(self.model_imgs_path) == False:
            print("Here")
            os.makedirs(self.model_imgs_path)

        # Build Model Architecture
        self._build()

    def _build(self):
        '''
        Function for creating all the models architectures
        ''' 
        self._build_encoder()
        self._build_sampler()
        self._build_decoder()
        self._build_full_model()
        
    def _build_encoder(self):
        '''
        Function for creating the encoder model architecture
        '''
        # INPUT
        self.encoder_input = Input(shape=self.encoder_input_dim, name='encoder_input')
        x = self.encoder_input

        # ARCHITECTURE: FULLY CONNECTED LAYERS
        for enc_hidden_layer in range(self.encoder_hidden_layers):
            x = Dense(self.encoder_hidden_layers_dim, activation=None, name='enc_dense_{}'.format(enc_hidden_layer))(x)

            if self.encoder_use_batch_norm: 
                x = BatchNormalization(name = 'enc_batch_norm_{}'.format(enc_hidden_layer))(x)

            x = ReLU(name = 'enc_relu_{}'.format(enc_hidden_layer))(x)

            if self.encoder_use_dropout:
                x = Dropout(rate = self.encoder_dropout_rate, name = 'enc_drop_{}'.format(enc_hidden_layer))(x)


        # Get Mu and Log Var vectors
        self.mu      = Dense(self.encoder_output_dim , activation=None, name='enc_mu')(x)
        self.log_var = Dense(self.encoder_output_dim , activation=None, name='enc_log_var')(x)

        # OUTPUTS
        self.encoder_output_mu      = self.mu
        self.encoder_output_log_var = self.log_var

        # MODEL ENCODER
        self.encoder = Model(inputs=(self.encoder_input), outputs=(self.encoder_output_mu, self.encoder_output_log_var), name='encoder')
        tf.keras.utils.plot_model(self.encoder,to_file='{}/encoder.png'.format(self.model_imgs_path), show_shapes=True, show_dtype=False, show_layer_names=True )

    def _build_sampler(self):
        '''
        Function for creating the sampling model architecture
        '''
        # INPUTS
        self.sampler_input_mu      = Input(shape=self.model_latent_dim, name='sampler_input_mu')
        self.sampler_input_log_var = Input(shape=self.model_latent_dim, name='sampler_input_log_var')

        # ARCHITECTURE
        def sampling(args):
            mu, log_var = args
            epsilon = keras_backend.random_normal(shape=[self.model_latent_dim], mean=0., stddev=1.)
            return mu + keras_backend.exp(log_var / 2) * epsilon  
        z = Lambda(sampling, name='sampler_z')([self.sampler_input_mu, self.sampler_input_log_var])

        # OUTPUTS
        self.sampler_output_z = z

        # MODEL DECODER
        self.sampler = Model(inputs=(self.sampler_input_mu, self.sampler_input_log_var), outputs=(self.sampler_output_z), name='sampler')
        tf.keras.utils.plot_model(self.sampler,to_file='{}/sampler.png'.format(self.model_imgs_path), show_shapes=True, show_dtype=False, show_layer_names=True )

    def _build_decoder(self):
        '''
        Function for creating the decoder model architecture
        '''
        # INPUT
        self.decoder_input_z = Input(shape=self.decoder_input_dim, name='decoder_input_z')
        x = self.decoder_input_z

        # ARCHITECTURE: FULLY CONNECTED LAYERS
        for dec_hidden_layer in range(self.decoder_hidden_layers):
            x = Dense(self.decoder_hidden_layers_dim, activation=None, name='dec_dense_{}'.format(dec_hidden_layer))(x)

            if self.decoder_use_batch_norm:
                x = BatchNormalization(name = 'dec_batch_norm_{}'.format(dec_hidden_layer))(x)

            x = ReLU(name = 'dec_relu{}'.format(dec_hidden_layer))(x)

            if self.decoder_use_dropout:
                x = Dropout(rate = self.decoder_dropout_rate, name = 'dec_drop_{}'.format(dec_hidden_layer))(x)


        # OUTPUT
        self.decoder_output = Dense(self.decoder_output_dim, activation="sigmoid", name='dec_output')(x)

        # MODEL DECODER
        self.decoder = Model(inputs=(self.decoder_input_z), outputs=(self.decoder_output), name='decoder')
        tf.keras.utils.plot_model(self.decoder,to_file='{}/decoder.png'.format(self.model_imgs_path), show_shapes=True, show_dtype=False, show_layer_names=True )

    def _build_full_model(self):
        '''
        Function for creating the full model architecture
        '''
        # INPUT
        self.model_input = self.encoder_input

        # ARCHITECTURE AND OUTPUTS
        (self.model_output) = self.decoder(self.sampler(self.encoder((self.model_input))))
        
        # MODEL FULL
        self.full_model = Model(inputs=(self.model_input), outputs=(self.model_output), name='full_model')


if __name__ == '__main__':
    print("\n\nWelcome to TimbreNet 3 Model")

    model = PianoRollModel( model_latent_dim=16,hidden_layers=3)