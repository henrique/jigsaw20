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
    ax = axs[0]
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='lower left')

#     # Plot training & validation accuracy values
#     ax = axs[1]
#     ax.plot(history.history['accuracy'])
#     ax.plot(history.history['val_accuracy'])
#     ax.set_title('Model accuracy')
#     ax.set_ylabel('Accuracy')
#     ax.set_xlabel('Epoch')
#     ax.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values
    ax = axs[1]
    ax.plot(history.history['auc'])
    ax.plot(history.history['val_auc'])
    ax.set_title('Model AUC')
    ax.set_ylabel('AUC')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    save_fig('history.png', path, bucket)
