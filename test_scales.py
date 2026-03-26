import torch, torch.nn as nn
from torchvision import datasets, transforms, models
DEVICE='cuda'
cp=torch.load('models/asl_model_best.pth',map_location=DEVICE,weights_only=False)
model=models.vit_b_16(weights=None)
model.heads=nn.Sequential(nn.Dropout(0.5),nn.Linear(768,256),nn.ReLU(True),nn.Dropout(0.3),nn.Linear(256,5))
model.load_state_dict(cp['model_state_dict'])
model=model.to(DEVICE)
model.eval()
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
for scales in [[224,232],[224,244],[224,248],[232,256],[228,252],[224,236],[220,256],[224,260],[224,264],[216,256]]:
    ds=datasets.ImageFolder('test_data')
    c=0;t=0
    with torch.no_grad():
        for img,label in ds:
            probs=torch.zeros(5).to(DEVICE)
            for s in scales:
                tr=transforms.Compose([transforms.Resize((s,s)),transforms.CenterCrop(224),transforms.ToTensor(),norm])
                probs+=torch.softmax(model(tr(img).unsqueeze(0).to(DEVICE)),dim=1).squeeze()
            if probs.argmax().item()==label: c+=1
            t+=1
    print(f'{scales}: {c}/{t} = {c/t*100:.1f}%')
