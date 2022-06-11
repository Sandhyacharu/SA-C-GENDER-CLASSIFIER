# SA-C-GENDER-CLASSIFIER
# Algorithm
1.Install deepface

2.Import necessary packages

3.Read the image

4.Analyze the gender using deepface

## Program:
```
/*
Program to implement Gender Classification
Developed by   : N Sandhya Charu
RegisterNumber :  212220230041
*/
#install deepface
pip install deepface

#import packages
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

#read the image
img=cv2.imread('Tvd.jpg')
plt.imshow(img[:,:,::-1])
plt.show()
img1=cv2.imread('tess.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()

#Analyze gender
result=DeepFace.analyze(img,actions=['gender'])
result1=DeepFace.analyze(img,actions=['emotion'])
result2=DeepFace.analyze(img1,actions=['gender'])
result3=DeepFace.analyze(img1,actions=['emotion'])

#print the gender and emotion
print("Gender : ",result['gender'])
print("Emotion : ",result1['emotion'])
print("Gender : ",result2['gender'])
print("Emotion : ",result3['emotion'])
```

## OUTPUT:
![image](https://user-images.githubusercontent.com/75235167/173190667-5c52729b-68ce-472f-ab6f-80803a2dc429.png)
![image](https://user-images.githubusercontent.com/75235167/173190676-c9c9ac5a-73b8-43c4-82d1-855f58c42323.png)


