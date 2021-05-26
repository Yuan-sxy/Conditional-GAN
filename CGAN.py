import jittor as jt
import argparse
import numpy as np
import math
import os
from jittor import nn
from jittor import init
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
from PIL import Image


parser = argparse.ArgumentParser(description='This is a implementation of a simple conditional GAN using jittor')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--input_vector_size', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--embedding_size', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--image_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channel', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
cmd_parse = parser.parse_args()

image_size=cmd_parse.image_size
channel=cmd_parse.channel
embedding_size=cmd_parse.embedding_size
image_shape=(channel,image_size,image_size)
batch_size=cmd_parse.batch_size
learning_rate=cmd_parse.learning_rate
b1=cmd_parse.b1
b2=cmd_parse.b2
epochs=cmd_parse.epochs
sample_interval=cmd_parse.sample_interval
random_input_vector_size=cmd_parse.input_vector_size

transform=transform.Compose([
        transform.Resize(image_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5],std=[0.5]),
    ])

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.label_embedding=nn.Embedding(embedding_size, embedding_size)
        self.model=nn.Sequential()
        # first layer (100+10,128)
        self.model.append(nn.Linear(random_input_vector_size+embedding_size,128))
        self.model.append(nn.LeakyReLU(0.2))
        # second layer
        self.model.append(nn.Linear(128,256))
        self.model.append(nn.LeakyReLU(0.2))
        # third layer
        self.model.append(nn.Linear(256,512))
        self.model.append(nn.LeakyReLU(0.2))
        # forth layer
        self.model.append(nn.Linear(512,1024))
        self.model.append(nn.LeakyReLU(0.2))
        # last layer
        self.model.append(nn.Linear(1024,int(np.prod(image_shape))))
        self.model.append(nn.Tanh())
        
    def execute(self,noise,label):
        # conditional GAN requires input which conbimes vector with corresponding labels
        input_data=jt.contrib.concat((self.label_embedding(label),noise),dim=1)
        # send input into netword and get result
        img=self.model(input_data)
        #change the result shape to image format
        img=img.view((img.shape[0],*image_shape))
        return img
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.label_embedding=nn.Embedding(embedding_size,embedding_size)
        self.model=nn.Sequential()
        # first layer
        self.model.append(nn.Linear(embedding_size+int(np.prod(image_shape)),512))
        self.model.append(nn.LeakyReLU(0.2))
        # second layer
        self.model.append(nn.Linear(512,512))
        self.model.append(nn.Dropout(0.4))
        self.model.append(nn.LeakyReLU(0.2))
        # last layer
        self.model.append(nn.Linear(512,1))
    
    def execute(self,img,label):
        # conditional GAN requires input which combines image with correspoding labels
        input_data=jt.contrib.concat((img.view((img.shape[0],(-1))),self.label_embedding(label)),dim=1)
        # sebd input data into network and get result
        d_result=self.model(input_data)
        return d_result
    

# get train data
def get_data():
    return MNIST(train=True,transform=transform).set_attrs(batch_size=batch_size,shuffle=True)


# change matrix data into image format and save it
def save_image(img,path,n_row=10,padding=5):
    # get parameter:N denotes batch_size,C denotes channel,W denotes width,H denotes height
    N,C,W,H=img.shape
    # N must be divide by n_row so it can generate a proper matrix
    if N%n_row!=0:
        print("N%n_row!=0")
        return
    # get column
    n_col=int(N/n_row)
    # to save all image
    img_all=[]
    # get image at matrix(i,j)
    for i in range(n_col):
        # img_ save the images which are in the same row
        img_=[]
        for j in range(n_row):
            # get the image at (i,j)
            img_.append(img[i*n_row+j])
            # get the initial image with zero padding
            img_.append(np.zeros((C,W,padding)))
        # np.concatenate() is used to concatenate two matrix
        # add images in the same row to img_all
        img_all.append(np.concatenate(img_,2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img=np.concatenate(img_all,1)
    img=np.concatenate([np.zeros((C,padding,img.shape[2])),img],1)
    img=np.concatenate([np.zeros((C,img.shape[1],padding)),img],2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img=img[:,:,::-1]
    elif C==1:
        img=img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row,batch_done):
    # generate n_row*n_row input z as a batch
    z=jt.array(np.random.normal(0,1,(n_row**2,random_input_vector_size))).float32().stop_grad()
    # generate label matrix as [[0,1,...,n_row-1],...,[0,1,...,n_row-1]]
    label=jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    # generate images through generator
    gen_img=generator(z,label)
    # save image generated by generator
    save_image(gen_img.numpy(),"images/%d.png"%batch_done,n_row=n_row)

# training
def train(data_loader,generator,discriminator,adversarial_loss):
    # define optimizer for Generator
    optimizer_G=nn.Adam(generator.parameters(),lr=learning_rate,betas=(b1,b2))
    # define optimizer for Discriminator
    optimizer_D=nn.Adam(discriminator.parameters(),lr=learning_rate,betas=(b1,b2))
    # begin training
    for epoch in range(epochs):
        for i,(img,label) in enumerate(data_loader):
            
            batch_size=img.shape[0]
            
            # adversarial ground truth
            valid=jt.ones([batch_size,1]).float32().stop_grad()
            fake=jt.zeros([batch_size,1]).float32().stop_grad()
            
            #input set
            real_img=jt.array(img)
            label=jt.array(label)
            
            
            # train Generator
            z=jt.array(np.random.normal(0,1,(batch_size,random_input_vector_size))).float32()
            gen_label=jt.array(np.random.randint(0,embedding_size,batch_size)).float32()
            
            gen_img=generator(z,gen_label)
            validity=discriminator(gen_img,gen_label)
            
            # G loss
            g_loss=adversarial_loss(validity,gen_label)
            g_loss.sync()
            optimizer_G.step(g_loss)
            
            # Train D
            valid_real=discriminator(real_img,label)
            d_real_loss=adversarial_loss(valid_real,valid)
            
            valid_fake=discriminator(gen_img.stop_grad(),gen_label)
            d_fake_loss=adversarial_loss(valid_fake,fake)
            
            d_loss=(d_real_loss+d_fake_loss)/2
            d_loss.sync()
            
            optimizer_D.step(d_loss)
            if i%50==0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss:%f] [G loss:%f]"%(epoch,epochs,i,len(data_loader),d_loss.data,g_loss.data))
            
            batch_done=epoch*len(data_loader)+i
            if batch_done%sample_interval==0:
                sample_image(n_row=10,batch_done=batch_done)
            
        if epoch%10==0:
            generator.save("saved_models/generator_last.pkl")
            discriminator.save("saved_models/discriminator_last.pkl")

if __name__=='__main__':
    # get instance of Generator & Discriminator
    generator=Generator()
    discriminator=Discriminator()
    
    # define loss function
    adversarial_loss=nn.MSELoss()

    os.makedirs("images",exist_ok=True)
    os.makedirs("saved_models",exist_ok=True)

    data_loader=get_data()
    
    train(data_loader,generator,discriminator,adversarial_loss)
