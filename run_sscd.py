from torchvision import transforms
import torch
from PIL import Image



normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])

results = []

folder2 = 'generated_images_mitigated'
folder1 = 'memorized_images'
device = 'cuda'
model = torch.jit.load("./models/sscd/sscd_disc_mixup.torchscript.pt").to(device)

for index in range(10):
    mem_name = "{}.jpg".format(str(index))

    try:
        img = Image.open(folder1+ "/" + mem_name).convert('RGB')
    except:
        continue
    batch = small_288(img).unsqueeze(0).to(device)
    emb = model(batch)[0, :]
    
    for j in range(3):
        
        if index < 10:
            img_name = "img_000{}_0{}.jpg".format(str(index), str(j))
        elif index < 100:
            img_name = "img_00{}_0{}.jpg".format(str(index), str(j))
        elif index < 1000:
            img_name = "img_0{}_0{}.jpg".format(str(index), str(j))
        else:
            img_name = "img_{}_0{}.jpg".format(str(index), str(j))
        img_name = "{}_{}.jpg".format(str(index), str(j))
        
        img2 = Image.open(folder2 + "/" + img_name).convert('RGB')
        batch2 = small_288(img2).unsqueeze(0).to(device)
        emb2 = model(batch2)[0, :]
        
        score = (emb-emb2).norm()
        print(score.item())
        results.append(score.item())

print(sum(results)/len(results))
            
    
