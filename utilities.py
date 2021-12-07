import numpy as np
from drive.MyDrive.kombiLab.rede.tf_bio import make_grid, rotate


def preprocess_features(
        features,
        coordinates,
        indices=None,
        rotation=0,
        grid_spacing=1.0,
        max_dist=10.0):
    """
    Rotates the coordinates and constructs the tensor grig.
    To extract a sub batch, use the indices parameter, e.g. indices=range(0,50) for the first 50 samples
    """

    if indices is None:  # grap all data if nothing specified
        indices = range(0, len(features))

    x = []
    for i, idx in enumerate(indices):
        coords_idx = rotate(coordinates[idx], rotation)
        features_idx = features[idx]
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=grid_spacing,
                 max_dist=max_dist))
    x = np.vstack(x)

    return x


def normalize_charge(data):
    """
    Normalize the partial charge attribute to stdev 1. Does not move the mean.
    """
    partial_charge_idx = 12 # TODO: This should be inside features or some vector, instead hard-coded for now

    charge_std = get_charge_std(data)
    data[..., partial_charge_idx] /= charge_std
    return data


def get_charge_std(features, partialcharge_idx=12):

    charges = []
    for feature_data in features: #['training']:
        # charges.append(feature_data[..., columns['partialcharge']])
        charges.append(feature_data[..., partialcharge_idx])

    charges = np.concatenate([c.flatten() for c in charges])

    charge_mean = charges.mean()
    charge_std = charges.std()
    #print('Partial charge mean=%s, sd=%s' % (charge_mean, charge_std))
    #print('use sd as scaling factor')

    return charge_std


def show_sample(features):
    '''
    This function prints out some features for two samples.
    Needs Featurizer to access the channel indices of the feature names.
    Not used in main code - keep for reference.
    '''

    #from tfbiodata import Featurizer
    #featurizer = Featurizer()
    #columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}
    columns = {'placeholder': 0}

    molcode_idx = 13  # this is columns['molcode'] - Featurizer
    assert ((features[:, :, :, :, molcode_idx] == 0.0).any()
            and (features[:, :, :, :, molcode_idx] == 1.0).any()
            and (features[:, :, :, :, molcode_idx] == -1.0).any()).all()

    # Get 2 samples: one with molcode 1.0 and one with -1.0 in this batch
    # This is just for demonstration
    idx1 = [[i[0]] for i in np.where(features[:, :, :, :, molcode_idx] == 1.0)]
    idx2 = [[i[0]] for i in np.where(features[:, :, :, :, molcode_idx] == -1.0)]

    print('\nexamples:')
    for mtype, mol in [['ligand', features[idx1]], ['protein', features[idx2]]]:
        print(' ', mtype)
        for name, num in columns.items():
            print('  ', name, mol[0, num])
            print('')
