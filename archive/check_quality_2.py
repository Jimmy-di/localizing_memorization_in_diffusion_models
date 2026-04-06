import os

for i in range(500):
    for j in range(5, 50):

        img_name = "./generated_images_blocked_cp/img_{}_{}.jpg".format(str(i), str(j))
        os.remove(img_name)