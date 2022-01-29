from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import io
import os
import json
import time
import pickle
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PianoRollModel():

    triad_structure_1   = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['j', 'n', 'a', 'd']), values=tf.constant([0, 0, 0, 0]),), default_value=tf.constant(-1))
    triad_structure_3   = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['j', 'n', 'a', 'd']), values=tf.constant([4, 3, 4, 3]),), default_value=tf.constant(-1))
    triad_structure_5   = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['j', 'n', 'a', 'd']), values=tf.constant([7, 7, 8, 6]),), default_value=tf.constant(-1))
    base_note_structure = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['Cn', 'Df', 'Dn', 'Ef', 'En', 'Fn', 'Gf', 'Gn', 'Af', 'An', 'Bf', 'Bn']), values=tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),), default_value=tf.constant(-1))
    volume_structure    = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['p', 'm', 'f']), values=tf.constant([0, 1, 2]),), default_value=tf.constant(-1))
    octave_structure    = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(['2', '3', '4']), values=tf.constant([3, 4, 5]),), default_value=tf.constant(-1))

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

    def pre_process_filename_to_roll(self, path):
        '''
        Fuction that preprocess  files to get pianorolls and metadata
        '''
        notes, volume = self.pre_process_filename_get_notes_volume_vectors(path)
        roll = self.pre_process_notes_volume_vectors_to_roll(notes, volume)
        return (roll), (roll)

    def pre_process_filename_get_notes_volume_vectors(self, path):
        '''
        Function that makes the metadata inputs (notes and olume) from the filename
        '''

        # Get metadata info from filename
        folders       = tf.strings.split(path, '/')
        file_name_wav = folders[-1]
        file_name     = tf.strings.split(file_name_wav,'.')[0]
        meta_data     = tf.strings.split(file_name, '_')
        octave        = meta_data[1]
        base_note     = meta_data[2]
        triad         = meta_data[3]
        volume_meta   = meta_data[4]

        # Make the 1/0 vectors for the model inputs
        # Notes
        first = tf.one_hot((self.octave_structure.lookup(octave)-3)*12+self.base_note_structure.lookup(base_note)+self.triad_structure_1.lookup(triad), 44)
        third = tf.one_hot((self.octave_structure.lookup(octave)-3)*12+self.base_note_structure.lookup(base_note)+self.triad_structure_3.lookup(triad), 44)
        fifth = tf.one_hot((self.octave_structure.lookup(octave)-3)*12+self.base_note_structure.lookup(base_note)+self.triad_structure_5.lookup(triad), 44)
        notes = first + third + fifth
        notes = tf.reshape(notes,[44])

        # Volumes
        volume = tf.one_hot(self.volume_structure.lookup(volume_meta), 3)
        volume = tf.reshape(volume,[3])

        return notes, volume

    def pre_process_notes_volume_vectors_to_roll(self, notes, volume):
        '''
        Function that makes the pianorrolls from the filename
        '''
        amplitude = ((tf.cast(tf.math.argmax(volume),tf.float32)+1)/3) # Piano is 0.33, metsoforte is 0.66 forte is 1
        pianoroll = notes * amplitude
        return pianoroll

    def load_weights(self, filepath):
        '''
        Function for loading weights from a trained model
        '''
        self.full_model.load_weights(filepath)

    def save_model_params(self, folder):
        '''
        Function that saves all the model parameters both in human readable text and machine readable text
        '''
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))
         
        # Machine readable save
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump(json.loads('"'+str(self.__dict__)+'"'), f)
        
        # Human readable save   
        with open(os.path.join(folder, 'params.txt'), 'w') as f:
            for key in self.__dict__.keys():
                f.write('{}: {}\n'.format(key, str(self.__dict__[key])))

    def compute_loss(self, train_data_point):
        '''
        Function that returns the total loss of the model
        '''
        train_input, y_true = train_data_point
        mu, log_var         = self.encoder(train_input)
        z                   = self.sampler((mu, log_var))
        y_pred              = self.decoder(z) 

        def vae_r_loss(y_true_r, y_pred_r):
            '''
            Fuction for specgram reconstruction losss
            '''
            r_loss = keras_backend.sum(keras_backend.square(y_true_r - y_pred_r), axis = [1])
            return r_loss

        def vae_kl_loss():
            '''
            Function for KL Divergence loss
            '''
            kl_loss =  -0.5 * keras_backend.sum(1 + log_var - keras_backend.square(mu) - keras_backend.exp(log_var), axis = 1)
            return kl_loss

        def vae_total_loss(y_true, y_pred):
            '''
            Function for calculating the pianorollloss including  reconstruction and KL Divergence
            '''
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss()
            total_loss = (self.r_loss_factor * r_loss) + kl_loss
            return  r_loss, kl_loss, total_loss

        r_loss, kl_loss, total_loss = vae_total_loss(y_true, y_pred)

        return total_loss, r_loss, kl_loss

    @tf.function
    def train_step(self, train_data_point, optimizer):
        """
        Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to update the model's parameters.
        """
        with tf.GradientTape() as tape:
            total_loss, r_loss, kl_loss = self.compute_loss(train_data_point)
        gradients = tape.gradient(total_loss, self.full_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.full_model.trainable_variables))
        return total_loss, r_loss, kl_loss

    def train_with_generator(self, train_data_flow, val_data_flow, epochs, initial_epoch, learning_rate, r_loss_factor, run_folder):
        '''
        Function for manually training the model
        '''
        # Save learning rate and loss weights as self variable
        self.learning_rate = learning_rate
        self.r_loss_factor = r_loss_factor

        # Create the optimizer object
        optimizer = Adam(learning_rate=self.learning_rate)

        # Create a variable for saving the best loss model
        prev_val_total_loss = 1e30

        # Callbacks Summary writers
        train_logdir_scalars = os.path.join(run_folder, "logs/scalars/train")
        train_file_writer_scalars = tf.summary.create_file_writer(train_logdir_scalars,filename_suffix='train')

        val_logdir_scalars = os.path.join(run_folder, "logs/scalars/val")
        val_file_writer_scalars = tf.summary.create_file_writer(val_logdir_scalars,filename_suffix='val')

        logdir_img = os.path.join(run_folder, "logs/image/")
        file_writer_img = tf.summary.create_file_writer(logdir_img)

        logdir_hist = os.path.join(run_folder, "logs/hist/")
        file_writer_hist = tf.summary.create_file_writer(logdir_hist)

        ETA_train_run = 0

        for epoch in range(1, epochs + 1):
            start_time_epoch = time.time()

            # TRAIN
            # Train metrics
            train_total_loss  = tf.keras.metrics.Mean()
            train_r_loss      = tf.keras.metrics.Mean()
            train_kl_loss     = tf.keras.metrics.Mean()

            # Progress bar init
            total_batches = len(train_data_flow)
            ETA_epoch_train = 0
            self.print_progress_bar(0, total_batches, prefix='\nTraning    epoch {}/{}:'.format(epoch, epochs), suffix='Complete.  Global epochs trained   {}/{}. ETA_EPOCH_TRAIN: {}. ETA_RUN: {}.'.format(epoch + initial_epoch, epochs + initial_epoch, str(datetime.timedelta(seconds=ETA_epoch_train)), str(datetime.timedelta(seconds=ETA_train_run))), length=50)

            # Train by iterating over batches
            for batch_number, train_data_point in enumerate(train_data_flow):
                start_time_batch= time.time()

                # Train
                total_loss, r_loss, kl_loss = self.train_step(train_data_point, optimizer)

                # Save losses in metrics
                train_total_loss(total_loss)
                train_r_loss(r_loss)
                train_kl_loss(kl_loss)

                # Update progress bar
                end_time_batch= time.time()
                ETA_epoch_train = int((end_time_batch - start_time_batch)*(total_batches - batch_number))
                self.print_progress_bar(batch_number + 1, total_batches, prefix='Traning    epoch {}/{}:'.format(epoch, epochs), suffix='Complete.  Global epochs trained   {}/{}. ETA_EPOCH_TRAIN: {}. ETA_RUN: {}.'.format(epoch + initial_epoch, epochs + initial_epoch, str(datetime.timedelta(seconds=ETA_epoch_train)), str(datetime.timedelta(seconds=ETA_train_run))), length=50)

            # VALIDATION
            # Validation metrics
            val_total_loss  = tf.keras.metrics.Mean()
            val_r_loss      = tf.keras.metrics.Mean()
            val_kl_loss     = tf.keras.metrics.Mean()

            # Progress bar init
            total_batches = len(val_data_flow)
            ETA_epoch_val = 0
            self.print_progress_bar(0, total_batches, prefix='\nValidating epoch {}/{}:'.format(epoch, epochs), suffix='Complete.  Global epochs validated   {}/{}. ETA_EPOCH_VAL: {}. ETA_RUN: {}.'.format(epoch + initial_epoch, epochs + initial_epoch, str(datetime.timedelta(seconds=ETA_epoch_val)), str(datetime.timedelta(seconds=ETA_train_run))), length=50)

            if epoch%10 == 0:  # Validate each 10 epochs for faster training 
                # Iterate over validation set
                for batch_number, val_data_point in enumerate(val_data_flow):
                    start_time_batch= time.time()

                    # Validate
                    total_loss, r_loss, kl_loss = self.compute_loss(val_data_point)

                    # Save losses in metrics
                    val_total_loss(total_loss)
                    val_r_loss(r_loss)
                    val_kl_loss(kl_loss)

                    # Update progress bar
                    end_time_batch= time.time()
                    ETA_epoch_val = int((end_time_batch - start_time_batch)*(total_batches - batch_number))
                    self.print_progress_bar(batch_number + 1, total_batches, prefix='Validating epoch {}/{}:'.format(epoch, epochs), suffix='Complete.  Global epochs validated   {}/{}. ETA_EPOCH_VAL: {}. ETA_RUN: {}.'.format(epoch + initial_epoch, epochs + initial_epoch, str(datetime.timedelta(seconds=ETA_epoch_val)), str(datetime.timedelta(seconds=ETA_train_run))), length=50)

                # CALLBACKS
                # Save model callback
                # Save model if it is better
                if (val_total_loss.result().numpy() < prev_val_total_loss):
                    prev_val_total_loss = val_total_loss.result().numpy()
                    self.full_model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))#, save_weights_only=True)
                    print('SAVED MODEL')

                # Scalars Callback
                with train_file_writer_scalars.as_default():
                    tf.summary.scalar('total_loss', train_total_loss.result(), step=epoch + initial_epoch)
                    tf.summary.scalar('r_loss', train_r_loss.result(), step=epoch + initial_epoch)
                    tf.summary.scalar('kl_loss', train_kl_loss.result(), step=epoch + initial_epoch)

                with val_file_writer_scalars.as_default():
                    tf.summary.scalar('total_loss', val_total_loss.result(), step=epoch + initial_epoch)
                    tf.summary.scalar('r_loss', val_r_loss.result(), step=epoch + initial_epoch)
                    tf.summary.scalar('kl_loss', val_kl_loss.result(), step=epoch + initial_epoch)

                # Image callback
                self.image_gen_callback(epoch + initial_epoch, val_data_flow, file_writer_img)
                
                # Histogram callback
                self.hist_gen(epoch + initial_epoch, val_data_flow, file_writer_hist)
                
                    

            # Print info for epoch
            end_time_epoch = time.time()
            ETA_train_run = int((end_time_epoch - start_time_epoch)*(epochs - epoch))
            print('Epoch: {}, t_total_loss: {:10.3f}, t_r_loss: {:10.3f}, t_kl_loss: {:10.3f}, v_total_loss: {:10.3f}, v_r_loss: {:10.3f}, v_kl_loss: {:10.3f},time elapsed: {}'.format(epoch, train_total_loss.result(),train_r_loss.result(),train_kl_loss.result(),val_total_loss.result(),val_r_loss.result(),val_kl_loss.result(),str(datetime.timedelta(seconds=(end_time_epoch - start_time_epoch)))))       

    def full_model_generate(self, input):
        '''
        Function for generating outputs from a trained model
        '''
        output = self.full_model.predict((input))
        return output_1

    def decoder_generate(self, z):
        '''
        Function for generating outputs from a trained model
        '''
        output = self.decoder.predict(z)
        return output

    def hist_gen(self, epoch, val_data_flow, file_writer_hist):
        z = self.sampler.predict(self.encoder.predict(val_data_flow))
        with file_writer_hist.as_default():
            for i in range(self.model_latent_dim):
                tf.summary.histogram("Latent_dim: "+str(i), z[:,i], step=epoch)

    def image_gen_callback(self, epoch, val_data_flow, file_writer_img):
        '''
        Function that crates th eimage and the audios for the audio - image callback
        '''
        n = 0
        image = []
        for example in val_data_flow:
            # Generate an example from random
            z = tf.random.normal(shape=(1,self.model_latent_dim,), mean=0.0, stddev=1)
            gen = self.decoder.predict(z)

            example_in, example_out = example
            orig = example_in
            recon = self.full_model.predict(example_in)

            image.append(self.plot_to_image(self.triple_out_to_plot(gen, orig, recon)))

            n = n+1
            if n == 9:
                break

        with file_writer_img.as_default():
            for i in range(9):
                tf.summary.image("Example "+str(i), image[i], step=epoch)

    def triple_out_to_plot(self, gen, orig, recon):
        '''
        Generates 3 plots with an generated example, an original and a reconstructed
        '''

        raw_gen   = self.visualize_roll(gen)
        raw_orig  = self.visualize_roll(orig)
        raw_recon = self.visualize_roll(recon)

        clean_gen   = self.visualize_roll(self.clean_pianoroll(gen))
        clean_orig  = self.visualize_roll(self.clean_pianoroll(orig))
        clean_recon = self.visualize_roll(self.clean_pianoroll(recon))


        # Generate plots
        fig, ax = plt.subplots(2, 3, sharey=True,figsize=(30,4))

        # Raw Gen
        p1 = ax[0,0].imshow(raw_gen, cmap='hot')
        p1_t = ax[0,0].title.set_text('Raw Generated')
        plt.colorbar(p1,ax=ax[0,0])

        # Clean Gen
        p2 = ax[1,0].imshow(clean_gen, cmap='hot')
        p2_t = ax[1,0].title.set_text('Clean Generated')
        plt.colorbar(p2,ax=ax[1,0])


        # Raw Original
        p3 = ax[0,1].imshow(raw_orig, cmap='hot')
        p3_t = ax[0,1].title.set_text('Raw Original')
        plt.colorbar(p3,ax=ax[0,1])

        # Clean Original
        p4 = ax[1,1].imshow(clean_orig, cmap='hot')
        p4_t = ax[1,1].title.set_text('Clean Original')
        plt.colorbar(p4,ax=ax[1,1])

        
        # Raw Reconstructed
        p5 = ax[0,2].imshow(raw_recon, cmap='hot')
        p5_t = ax[0,2].title.set_text('Raw Reconstructed')
        plt.colorbar(p5,ax=ax[0,2])

        # Clean Reconstructed
        p6 = ax[1,2].imshow(clean_recon, cmap='hot')
        p6_t = ax[1,2].title.set_text('Clean Reconstructed')
        plt.colorbar(p6,ax=ax[1,2])

        return fig

    def visualize_roll(self, pianoroll):
        '''
        Generates a piano over the pianorroll for visual reference
        '''
        visual_roll = np.repeat(pianoroll, repeats=3, axis=0)
        visual_roll = np.append(np.zeros((1,44)),visual_roll,axis=0)
        visual_roll = np.append((np.array([[1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1]]))*-1+1,visual_roll,axis=0)
        return visual_roll


    def clean_pianoroll(self, raw_pianoroll):
        '''
        rounds the continous volume value to the closest volume.
        '''
        clean_pianoroll = np.floor(3*raw_pianoroll+0.5)/3
        return clean_pianoroll

    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def print_progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'X', printEnd = "\r"):
        '''
        Fuction that prints a progress bar
        '''
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

if __name__ == '__main__':
    print("\n\nWelcome to TimbreNet 3 Model")

    model = PianoRollModel( model_latent_dim=16, hidden_layers=2, hidden_layers_dim=32)