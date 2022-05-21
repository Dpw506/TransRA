from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test=[]
        for line in open(os.path.join(test_dir, 'val.txt')):
            line = line.strip('\n')
            if line!='':
                self.list_test.append(line)
        # self.root_hazy = os.path.join(test_dir , 'hazy/')
        # self.root_clean = os.path.join(test_dir , 'clean/')
        self.root_hazy = os.path.join(test_dir, 'input/')
        self.root_clean = os.path.join(test_dir , 'target/')
        self.file_len = len(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy + self.list_test[index])
        clean = Image.open(self.root_clean + self.list_test[index])
        hazy = self.transform(hazy)

        hazy_up=hazy[:,0:256,:]
        hazy_upl = hazy_up[:,:,0:256]
        hazy_upr = hazy_up[:,:,256:512]
        hazy_down=hazy[:,256:512,:]
        hazy_downl = hazy_down[:,:,0:256]
        hazy_downr = hazy_down[:,:, 256:512]
        clean = self.transform(clean)
        return hazy_upl,hazy_upr,hazy_downl,hazy_downr,hazy,clean

    def __len__(self):
        return self.file_len





