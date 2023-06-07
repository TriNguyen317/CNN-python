import numpy as np
import mnist
from Convolution import *
from MaxPooling import *
from Softmax import *
from tqdm import tqdm



class CNN:
    def __init__(self, x_train, y_train) :
        self.X = x_train
        self.Y = y_train
        
    def forward(self,X_train, Y_train):
        '''
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = conv.forward((X_train / 255))
        out = pool.forward(out)
        out = softmax.forward(out)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[Y_train])
        acc = 1 if np.argmax(out) == Y_train else 0
        result = np.argmax(out)

        return out, loss, acc, result


    def train(self, X_train, Y_train, lr=.005):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Forward
        out, loss, acc, _ = self.forward(X_train, Y_train)

        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[Y_train] = -1 / out[Y_train]

        # Backprop
        gradient = softmax.backprop(gradient, lr)
        gradient = pool.backprop(gradient)
        gradient = conv.backprop(gradient, lr)

        return loss, acc


    def fit(self, epochs=10):
        print("---------Training-------------")
    # Train the CNN for 3 epochs
        for epoch in range(epochs):

            # Shuffle the training data
            permutation = np.random.permutation(len(self.X))
            train_images = self.X[permutation]
            train_labels = self.Y[permutation]

            # Train!
            
            print('--- Epoch %d ---' % (epoch + 1))
            loss = 0
            num_correct = 0
            with tqdm(total=len(train_images), leave=False, desc=f"Epoch {epoch+1}") as pbar:
                for im, label in zip(train_images, train_labels):
                    
                    l, acc = self.train(im, label)
                    loss += l
                    num_correct += acc

                    pbar.set_postfix(Average_Loss=loss/100, Accuracy=num_correct)
                    pbar.update(1)
                print()
    
    def test(self,x_test,y_test):
        y_pred=[]
        num_correct=0
        for x,y in zip(x_test,y_test):
            _,_,accuracy,pred=self.forward(x,y)
            y_pred.append(pred)
            num_correct+=accuracy
        num_tests = len(test_images)
        print('Test Accuracy:', num_correct / num_tests)
        
            
            
            
# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
# for im, label in zip(test_images, test_labels):
#     _, l, acc = forward(im, label)
#     loss += l
#     num_correct += acc
# Use 0->1000 to test and 1000->3000 to train
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
train_images = mnist.test_images()[1000:1500]
train_labels = mnist.test_labels()[1000:1500]
conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10
model=CNN(train_images, train_labels)
model.fit()
model.test(test_images,test_labels)
