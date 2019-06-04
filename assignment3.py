import cv2
import numpy
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt

def get_features(path1,vec):
    '''This function goes to path1, reads all the images present there,
    expands the image into a 1D array,appends all the feature vectors 
    and stores them in vec. It also returns the shape of each image'''
    listing = os.listdir(path1)
    for file in listing:
        img = cv2.imread(path1 + '/' + file)
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        size=img.shape
        arr=numpy.reshape(img, (numpy.product(img.shape),))
        vec.append(arr)  
    return size

 

def get_filepaths(directory,vec):
    """
    This function will generate the file names in a directory. Used to get the paths
    of folders which contain the images.
    """
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for dirname in directories:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, dirname)
            size = get_features(filepath,vec)
    return size

def reconstruction(num,img,U,size,m):
    '''This function reconstructs the given image using num number of eigenfaces'''
    
    img1=img-m
    sum1=numpy.zeros(size[0]*size[1])
    sm=numpy.zeros(size)
    for i in range(num):
        v=numpy.dot(img1,U[:,i])
        im=numpy.multiply(v,U[:,i])
        sum1=sum1+im
    sum1=sum1+m
    sm=numpy.reshape(sum1, size)
    return sm   
    
X=[]
size=get_filepaths('gallery',X)

m=numpy.mean(X, 0)
X -= m
v1=numpy.zeros((200,10304))
S=(1.0/200)*numpy.dot(X,X.T)

evalues, evectors = LA.eigh(S)
'Sort the eigen values and get the required order of indices'

idx = evalues.argsort()[::-1]
evectors = evectors[:,idx]
evalues = evalues[idx]

U = numpy.dot(X.T,evectors)
'Normalization of eigenvectors and storing eigenfaces'
for i in range(200):
    norm=numpy.linalg.norm(U[:,i])
    U[:,i]=U[:,i]/norm
    img=numpy.reshape(U[:,i], size)
    img=img-img.min()
    img=(img/img.max())*255
    p=r'eigenfaces\im'+str(i)+'.jpg'
    cv2.imwrite(p,img.astype(numpy.uint8))
   
a=100*numpy.cumsum(evalues)/max(numpy.cumsum(evalues))
plt.plot(a)
plt.ylabel('Percent of variance captured')
plt.xlabel('eigenvalues')  
plt.title('Graph showing percent of variance captured by varying the number of eigenvectors') 
plt.grid(True)
plt.show()

print('eigenvalues capture 95% of the variance:')
for i in range(200):
    if a[i]>=95:
        print(i+1)
        break

'For reconstruction of image1'
img = cv2.imread('face_input_1.pgm')        
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
size=img.shape
img1=numpy.reshape(img, (numpy.product(img.shape),))

cv2.imwrite('Face1_0.jpg',reconstruction(1,img1,U,size,m).astype(numpy.uint8)) 
cv2.imwrite('Face1_15.jpg',reconstruction(15,img1,U,size,m).astype(numpy.uint8)) 
cv2.imwrite('Face1_200.jpg',reconstruction(200,img1,U,size,m).astype(numpy.uint8)) 

err=[]
for i in range(200):
	diff=img-reconstruction(i,img1,U,size,m)
	sq=sum(sum(numpy.square(diff)))/10304
	err.append(sq)
 
plt.plot(err)
plt.ylabel('Mean Squared error ')
plt.xlabel('Number of eigenvalues used')  
plt.title('Mean Squared error of input image 1 as a function of number of eigenvectors used') 
plt.grid(True)
plt.show()

'For reconstruction of image2'
img = cv2.imread('face_input_2.pgm')        
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
size=img.shape

img1=numpy.reshape(img, (numpy.product(img.shape),))

cv2.imwrite('Face2_0.jpg',reconstruction(1,img1,U,size,m).astype(numpy.uint8)) 
cv2.imwrite('Face2_15.jpg',reconstruction(15,img1,U,size,m).astype(numpy.uint8)) 
cv2.imwrite('Face2_200.jpg',reconstruction(200,img1,U,size,m).astype(numpy.uint8)) 

err=[]
for i in range(200):
	diff=img-reconstruction(i,img1,U,size,m)
	sq=sum(sum(numpy.square(diff)))/10304
	err.append(sq)
 
plt.plot(err)
plt.ylabel('Mean Squared error ')
plt.xlabel('Number of eigenvalues used')  
plt.title('Mean Squared error of input image 2 as a function of number of eigenvectors used') 
plt.grid(True)
plt.show()