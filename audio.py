import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import fnmatch
import os
import itertools



class PLOTS:
    """
    Clase PLOTS que reúne los métodos para plotear todo lo necesario en el notebook
    """
    def __init__(self):
        """
        Constructor de la clase
        """
        print("Instancia de la clase PLOTS creada")
    
    def plot_spec_DataAug(self, data: np.array, sr: int, title: str, fpath: str) -> None:
        """
        Función que realiza un plot del espectograma y la forma de onda del audio. 
        Utilizada para ver los resultados obtenidos tras aplicar Data augmentation.

        Args:
            data (np.array): Audio en forma de np array.
            sr (int): Sampling rate.
            title (str): Título que se quiere añadir y será modificado al final con el tipo de audio a plotear.
            fpath (str): Path al audio.
        """
        label = str(fpath).split('/')[-1].split('_')[0]
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].title.set_text(f'{title} / Label: {label}')
        ax[0].specgram(data,Fs=2)
        fig.savefig('temp1.png',bbox_inches="tight",pad_inches=0)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0,1,len(data)),data)


    def plot_audio(self, audio: str = 'archive/set_a/normal__201106111136.wav', plot_type: str = 'spectogram', audio_type: str = 'NORMAL'):
        """
        Función que nos van a permitir realizar el espectograma, la forma de onda y el MFCC.


        Args:
            audio (str, optional): Path al audio. Defaults to 'archive/set_a/normal__201106111136.wav'.
            plot_type (str, optional): Tipo de Gráfico que queremos ver: posibilidades -> spectogram, waveform, MFCC. Defaults to 'spectogram'.
            audio_type (str, optional): [description]. Defaults to 'NORMAL'.
        """
        y, sr = librosa.load(audio) 
        dur = librosa.get_duration(y)
        print('El sampling rate por defecto es: {}'.format(sr))
        print ('La duración del audio es: {}'.format(dur))
        print('El tamaño de y (con las amplitudes correspondientes de la onda en t) es: {}'.format(y.shape))
        plt.figure(figsize=(18,5))
            
        if plot_type == 'waveform':
            librosa.display.waveplot(y, sr=sr)
            plt.title('Waveform - {}'.format(audio_type))
        
        elif plot_type == 'spectogram':
            spect = np.abs(librosa.stft(y))
            librosa.display.specshow(librosa.amplitude_to_db(spect, ref=np.max),x_axis='time', y_axis='log')
            plt.title('Power spectrogram - {}'.format(audio_type))
            plt.plot()
        
        else:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time') #from the literature the first 13th are the most representative
            print('Las dimensiones del MFCC son {} lo que implica que hay {} muestras y {} coeficientes de MFCC'.format(mfccs.shape, mfccs.shape[0], mfccs.shape[1])) #That's 388 samples and 13 Mel-frequency cepstral coefficients
            plt.title('MFCCS - {}'.format(audio_type))
            plt.colorbar()
            plt.plot()

    def plot_history(self, history):
        """
        Plot Keras History
        Plot loss and accuracy for the training and validation set.

        Args:
            history ([type]): history.
        """ 
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        plt.figure(figsize=(22,10))
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        ## Accuracy
        plt.figure(221, figsize=(20,10))
        ## Accuracy
        # plt.figure(2,figsize=(14,5))
        plt.subplot(221, title='Accuracy')
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        ## Loss
        plt.subplot(222, title='Loss')
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))    
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def plot_confusion_matrix(self, cm, classes, cmap, normalize: bool = False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Args:
            cm ([type]): confusion matrix.
            classes ([type]): classes.
            normalize (bool, optional): Normalize. Defaults to False.
            cmap ([type], optional): Colors map. Defaults to plt.cm.Blues.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


class UTILS:
    """
    Clase PLOTS que reúne los métodos para plotear todo lo necesario en el notebook
    """
    def __init__(self):
        """
        Constructor de la clase
        """
        print("Instancia de la clase UTILS creada")


    def create_labels(self, classes: list=['artifact','abnormal','normal']):
        """
        Creation of the dictionary for assigning to each category its corresponding label

        Args:
            classes (list, optional): Crea las labels. Defaults to ['artifact','murmur','normal'].

        Returns:
            [type]: maps de labels y array de clases.
        """
        label_to_int = {k:v for v,k in enumerate(classes)}
        # map integer to label text
        int_to_label = {v:k for k,v in label_to_int.items()}
        print ('The labels assigned to each category are set as follows {}'.format(label_to_int))
        print('The other way around would be {}'.format(int_to_label))
        return label_to_int, int_to_label,classes


    def select_label_class(self, file_name: str) -> int:
        """
        Función que machea los distintos labels

        Args:
            file_name (str): Path de la ruta del fichero

        Returns:
            int: 
        """
        if "artifact" in file_name:
            return 0
        elif "normal" in file_name:
            return 2
        elif ("Aunlabelledtest" in file_name or "Bunlabelledtest" in file_name):
            return -1
        else:
            return 1


    def extract_features(self, file_name, max_pad_len: int = 586, cnn: bool = False, duration: int = 12):
        """
        Función que extrae los parámetros del audio que introduciremos al modelo.

        Args:
            file_name ([type]): Ruta al audio del que se quiere extraer información.
            max_pad_len (int, optional): . Defaults to 600.
            cnn (bool, optional): [description]. Defaults to False.

        Args:
            file_name (str): Ruta al audio del que se quiere extraer información.
            max_pad_len (int, optional): Padding. Defaults to 586.
            cnn (bool, optional): Extracción de features para una CNN. Defaults to False.
            duration (int, optional): Máxima duración de los audios. Defaults to 12.

        Returns:
            [type]: mfccs.
        """
        try:
            if cnn == True:
                audio, sample_rate = librosa.load(file_name, duration=duration,res_type='kaiser_fast')
                dur = librosa.get_duration(y=audio, sr=sample_rate)

                # # pad audio file same duration
                if (round(dur) > duration):
                    print ("fixing audio lenght :", file_name)
                    audio = librosa.util.fix_length(audio, duration)

                # audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfccs = np.mean(mfccs.T,axis=0)
            
        except:
            print("Error encountered while parsing file: ", file_name)
            return None 
        
        return mfccs