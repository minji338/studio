from keras.callbacks import Callback

class BrighticsLogger(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        super(BrighticsLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        log_labels = ['epoch'] + self.params['metrics']
        self.log_format = ','.join(['%s'] * len(log_labels)) + '\n'

        self.write_log(self.log_format % tuple(log_labels))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_values = [str(epoch + 1)]
        for k in self.params['metrics']:
            if k in logs:
                log_values.append(str(logs[k]))
            else:
                log_values.append('NaN')

        self.write_log(self.log_format % tuple(log_values))

    def on_train_end(self, logs=None):
        self.write_log('END\n')

    def write_log(self, log):
        with open(self.filepath, 'a') as log_file:
            log_file.write(log)
