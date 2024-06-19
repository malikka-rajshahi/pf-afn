import requests as rq
import numpy as np
import time
from zipfile import ZipFile
import psutil

import cv2
from requests.api import post
# URL = "http://192.168.0.101:8080" 
# URL =  "http://clothestryonflaskapp-env.eba-4tddjvnu.ap-south-1.elasticbeanstalk.com" # "http://127.0.0.1:8000"
# URL = "http://127.0.0.1:80"
#URL = "http://3.7.252.211:80" # aws ecs creds

#URL ="http://localhost:9000/2015-03-31/functions/function/invocations"
URL= 'http://localhost:9000/send_here'
start = time.time()
files = {
    "real_image": open("./dataset/test_img/clothes_1.jpg","rb"),
    "clothes": open("./dataset/test_clothes/clothes_test_1.jpg","rb"),
    "edge": open("./dataset/test_edge/clothes_test_1.jpg","rb")
}
# print('The CPU usage is: ', psutil.cpu_percent(6))
# print('RAM memory % used:', psutil.virtual_memory()[2])

post_r = rq.post(url=URL,files=files)

print(post_r.content)

open('op.jpg', 'wb').write(post_r.content)

end = time.time()
print("time taken: "+str(end-start))

