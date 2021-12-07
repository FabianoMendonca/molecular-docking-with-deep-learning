#from drive.MyDrive.kombiLab.rede.utilities import preprocess_features
from drive.MyDrive.kombiLab.rede.tf_bio import make_grid, rotate
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,list_id, list_features, coords, list_affinity, batch_size=32,std = 0.42584962, dim=(21,21,21), n_channels=19, shuffle= False, on_training = True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_id = list_id
        self.labels = list_affinity
        self.list_features = list_features
        self.coords = coords
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.std = std
        self.rotation = 0
        self.step = 0
        self.on_training = on_training
        self.on_epoch_end()
        self.features_names = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode', 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
        self.columns = {name: i for i, name in enumerate(self.features_names)}

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.list_id) / self.batch_size))*25)

    def __getitem__(self, index):
        'Generate one batch of data'
        # 

        if(self.on_training and (self.rotation==0) and (self.step==0)):
          print('\n Rotação: ', self.rotation, '\n')
        if(self.step > (len(self.list_id) / self.batch_size)):
          self.step = 0
          self.rotation = self.rotation + 1
          self.indexes = np.arange(len(self.list_features))
          if(self.on_training):
            print('\n Rotação: ', self.rotation, '\n')

        # Generate indexes of the batch

        #self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes = self.indexes[self.step*self.batch_size:(self.step+1)*self.batch_size]

        #indexes = range (self.step*self.batch_size , ((self.step+1)*self.batch_size)-1)

        # Find list of IDs
        list_IDs_temp = [self.list_id[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, self.rotation)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.rotation = 0
        self.step = 0
        self.indexes = np.arange(len(self.list_features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, rotation ):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype='float32')
        self.step = self.step + 1  
        #steps per epoch == len(list_id)/batch_size

        # Generate data ( get_batch )

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            tmp_index = self.list_id.index(ID)
            if (rotation>0):
              tmp_coords = rotate(self.coords[tmp_index], (rotation-1))
              X[i] = make_grid(tmp_coords, self.list_features[tmp_index])
            else:
              X[i] = make_grid(self.coords[tmp_index], self.list_features[tmp_index])
            

            # Store class
            #print(ID)
            y[i] =self.labels[tmp_index]
        X[..., self.columns['partialcharge']] /= self.std
  
        return X, y
