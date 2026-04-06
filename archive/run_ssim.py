import PIL
from PIL import Image

from rembg import new_session, remove

import argparse
from skimage.metrics import structural_similarity
import numpy as np

def fb_strip(session, im):
    foreground = remove(im,
        session=session, 
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20, 
        alpha_matting_erode_size=11)
    
    alpha_chan = foreground.split()[-1].convert("L")
    #inverted_alpha = PIL.ImageOps.invert(alpha)

    background = im.copy().paste(alpha_chan, mask=alpha_chan)
    
    return foreground, background

def calculate_score(image1, image2):
    image1 = image1.resize((512,512))
    image2 = image2.resize((512,512))
    img1 = np.asarray(image1)#.convert("L"))
    img2 = np.asarray(image2)#.convert("L"))
    
    #print(img1.shape)  # prints (30, 60, 3)
    #print(img2.shape)  # prints (30, 60, 3)
     
    measure_value = structural_similarity(img1, img2, win_size=3) #, multichannel=True)
    return measure_value

def main(args):

    model_name = "birefnet-general"
    session = new_session(model_name)

    mem_folder = "./memorized_images/"
    gen_folder = args.sname

    num_images = 500

    scores = []

    for i in range(num_images):

        print(str(i) + " / " + str(num_images))

        im_name = "{}.jpg".format(str(i))
        mem_image = Image.open(mem_folder + im_name)
        mem_foreground, mem_background = fb_strip(session, mem_image)



        for j in range(0, 5):

            if i < 10:
                img_name = "img_000{}_0{}.jpg".format(str(i), str(j))
            elif i < 100:
                img_name = "img_00{}_0{}.jpg".format(str(i), str(j))
            elif i < 1000:
                img_name = "img_0{}_0{}.jpg".format(str(i), str(j))
            else:
                img_name = "img_{}_0{}.jpg".format(str(i), str(j))
            
            #img_name = "{}_{}.jpg".format(str(index), str(second_i))

            gen_img = Image.open(gen_folder + '/' + img_name)

            gen_foreground, gen_background = fb_strip(session, gen_img)

            score = 0.5*calculate_score(mem_foreground, gen_foreground) + 0.5*calculate_score(mem_background, gen_background)
        
            print(score)
            scores.append(score)
    print(scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--sname', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()

    main(args)