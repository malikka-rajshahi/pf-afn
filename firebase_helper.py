import firebase_admin
from firebase_admin import credentials, storage
import datetime
import requests
from requests.api import get
import os

class FirebaseHelper:

    def __init__(self):
        self.cred = credentials.Certificate("tvish_firebase_key.json")
        if not firebase_admin._apps:
            self.app = firebase_admin.initialize_app(self.cred, {
            'storageBucket': 'tvish-ecommerce.appspot.com',
		},name='tvish-storage-manager')
        else:
            self.app = firebase_admin.get_app(name='tvish-storage-manager')
        self.strg = storage.bucket(app=self.app)

    
    def getImageromGSUrl(self,blobPath,imgType=0):
        '''
            `imgType` has two values; 0 and 1,
            0 is for cloth image type and
            1 is for cloth edge image type
        '''
        if blobPath == '' or blobPath == None:
            raise('No storage blob path found!')
            
        blobPath = blobPath.split(".com/")[-1]
        if imgType == 0:
            
            #os.mkdir('tmp/test_clothes')
            

            storageLoc = "/tmp/test_clothes/clothes_test_1.jpg"
        else:
            #try:
            #	os.mkdir('tmp/test_edge')
            #except:
            #	pass
            storageLoc = "/tmp/test_edge/clothes_test_1.jpg"
        
        self.blob = self.strg.blob(blobPath)
        genUrl = self.blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        response = requests.get(genUrl)
        # print(response.content)
        with open(storageLoc,'wb') as writeFile:
            writeFile.write(response.content)

if __name__ == '__main__':
    obj = FirebaseHelper()
    # edge -- new_product_images/edges/0.png
    # cloth -- new_product_images/clothes/0.jpg
    obj.getImageromGSUrl(blobPath="new_product_images/clothes/1.jpg")
    obj.getImageromGSUrl(blobPath="new_product_images/edges/1.png",imgType=1)
