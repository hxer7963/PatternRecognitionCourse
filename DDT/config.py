import warnings

class DefaultConfig(object):
    env = 'default'
    data_root = './data/'
    
    pca_model_pth = 'checkpoint/pca_model.pt'
    cuda = True 
    batch_size = 128
    num_workers = 20
    print_freq = 20

    
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' % k)
            setattr(self, k, v)

    def print_config(self):    
        print('Configurate: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse':
                print(k, getattr(self, k))
        import torch
        if not torch.cuda.is_available():
            self.cuda = False

opt = DefaultConfig()