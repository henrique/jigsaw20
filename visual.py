""" plot and save to gcs """
from matplotlib import pyplot as plt
from google.cloud import storage

def save_fig(filename, path, gcs):
    """ save current plt fig to gcs """
    plt.gcf().savefig(filename)
    plt.close()
    # init GCS client and upload file
    client = storage.Client()
    bucket = client.get_bucket(gcs)
    blob = bucket.blob(f'{path}/{filename}')
    blob.upload_from_filename(filename=filename)


def plot_history(history, path, bucket):
    """ plot a save the model's history """
    ## Eval
    _, axs = plt.subplots(1, 2, figsize=(18, 4))
    # Plot training & validation loss values
    ax_ = axs[0]
    ax_.plot(history.history['loss'])
    ax_.plot(history.history['val_loss'])
    ax_.set_title('Model loss')
    ax_.set_yscale('log')
    ax_.set_ylabel('Loss')
    ax_.set_xlabel('Epoch')
    ax_.legend(['Train', 'Test'], loc='lower left')

#     # Plot training & validation accuracy values
#     ax_ = axs[1]
#     ax_.plot(history.history['accuracy'])
#     ax_.plot(history.history['val_accuracy'])
#     ax_.set_title('Model accuracy')
#     ax_.set_ylabel('Accuracy')
#     ax_.set_xlabel('Epoch')
#     ax_.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values
    ax_ = axs[1]
    ax_.plot(history.history['auc'])
    ax_.plot(history.history['val_auc'])
    ax_.set_title('Model AUC')
    ax_.set_ylabel('AUC')
    ax_.set_xlabel('Epoch')
    ax_.legend(['Train', 'Test'], loc='upper left')

    save_fig('history.png', path, bucket)
