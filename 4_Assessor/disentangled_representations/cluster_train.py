import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()


from pipeline import train_and_validate

# train_and_validate(
#     dataset_name = 'zalando_germany',
#     batch_size= 256,
#     dcor_loss_factor=0,
#     embeddings_name='vits14',
#     grl_weight=None,
#     hidden_dims_branches=[128, 128, 32],
#     hidden_dims_common=[256, 256],
#     lr=0.001,
#     max_epochs=100,
#     prediction_loss_factor=1,
#     seed=4243
# )


train_and_validate(
    dataset_name = 'zalando_germany',
    batch_size= 256,
    dcor_loss_factor=9.670528789445637,
    embeddings_name='vitb14',
    grl_weight=None,
    hidden_dims_branches=[256, 256, 256],
    hidden_dims_common=[256, 256],
    lr=0.001,
    max_epochs=100,
    prediction_loss_factor=1,
    seed=33
)