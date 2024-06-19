import requests as rq
# import numpy as np
import time
from zipfile import ZipFile
# import psutil
import base64

# import cv2
from requests.api import post
# URL = "http://192.168.0.101:8080"
# URL =  "http://clothestryonflaskapp-env.eba-4tddjvnu.ap-south-1.elasticbeanstalk.com" # "http://127.0.0.1:8000"
# URL = "http://127.0.0.1:80"
#URL = "http://localhost:9000/2015-03-31/functions/function/invocations"  # aws ecs creds
#URL= 'https://b5260kwkp0.execute-api.ap-south-1.amazonaws.com/beta-test/tryon'
#URL='https://2svoae9ip2.execute-api.ap-south-1.amazonaws.com/Prod/classify_digit/'
# URL = 'http://127.0.0.1:3000/clothes_tryon'
#URL = 'https://2svoae9ip2.execute-api.ap-south-1.amazonaws.com/Prod/classify_digit/'
URL = "https://f54btqwyv9.execute-api.ap-south-1.amazonaws.com/Prod/clothes_tryon/"
start = time.time()
b64_ims = []
#l=os.listdir(r'C:\Users\akash\FILES\Internship\gen_test\application\dataset\test_img')
# for i in l:
#     files = {
#         "real_image": open("./dataset/test_img/"+i,"rb"),
#         "clothes": open("./dataset/test_clothes/clothes_test_1.jpg","rb"),
#         "edge": open("./dataset/test_edge/clothes_test_1.jpg","rb")
#     }
    # print('The CPU usage is: ', psutil.cpu_percent(6))
    # print('RAM memory % used:', psutil.virtual_memory()[2])

with open("dataset/test_img/5.jpg", "rb") as img_file:
    #my_string = base64.b64encode(img_file.read())
    my_string =base64.b64encode(img_file.read()).decode("utf-8")
# print(my_string)
with open("dataset/test_clothes/clothes_test_1.jpg","rb") as cloth_file:
    cloth_str = base64.b64encode(cloth_file.read()).decode("utf-8")

with open("dataset/test_edge/clothes_test_1.jpg","rb") as edge_file:
    edge_str = base64.b64encode(edge_file.read()).decode("utf-8")
data = {
    "real_image": my_string,
    "cloth_bytes": cloth_str,
    "edge_bytes": edge_str,
    "cloth_image": "new_product_images/clothes/5.jpg",
    "edge_image": "new_product_images/edges/5.jpg"
}
#print(type(my_string))
post_r = rq.post(url=URL,json=data)

print(post_r.content)

open('op-234_server.jpg', 'wb').write(base64.b64decode(post_r.content))
end=time.time()
print('total time ',end-start)

