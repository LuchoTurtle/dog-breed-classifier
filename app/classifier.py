import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from PIL import Image
from torchvision import transforms


###################
# DETECT THE FACE #
###################

def face_detector(path):
    cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

    pil_image = Image.open(path).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    grayscale = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    detected_faces = cascade.detectMultiScale(grayscale)

    return len(detected_faces) > 0

##################################
# DETECT IF IT IS A CAT OR A DOG #
##################################

model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model_resnet34 = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)

for name, param in model_resnet18.named_parameters():
    if ("bn" not in name):
        param.requires_grad = False

for name, param in model_resnet34.named_parameters():
    if ("bn" not in name):
        param.requires_grad = False


num_classes = 2

model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))

model_resnet34.fc = nn.Sequential(nn.Linear(model_resnet34.fc.in_features,512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))

batch_size=32
img_dimensions = 224

# Normalize to the ImageNet mean and standard deviation
# Could calculate it for the cats/dogs data set, but the ImageNet
# values give acceptable results here.
img_transforms = transforms.Compose([
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

img_test_transforms = transforms.Compose([
    transforms.Resize((img_dimensions,img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

resnet18 = torch.hub.load('pytorch/vision', 'resnet18')
resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
resnet18.load_state_dict(torch.load('./model_resnet18.pth'))
resnet18.eval()

resnet34 = torch.hub.load('pytorch/vision', 'resnet34')
resnet34.fc = nn.Sequential(nn.Linear(resnet34.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
resnet34.load_state_dict(torch.load('./model_resnet34.pth'))
resnet34.eval()

models_ensemble = [resnet18.to(device), resnet34.to(device)]


def dog_or_cat_predict(img_path):
    image = Image.open(img_path)

    # These are the transforms according to the documentation in pytorch.models (https://pytorch.org/hub/pytorch_vision_vgg/)
    transform = transforms.Compose([transforms.Resize(img_dimensions),
                                    transforms.CenterCrop(img_dimensions),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # Apply transforms and add new dimension (to input in the network)
    image_tensor = transform(image).unsqueeze(0)

    # move the input and model to GPU
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions = [i(image_tensor).data for i in models_ensemble]
        avg_predictions = torch.mean(torch.stack(predictions), dim=0)
        _, predicted = torch.max(avg_predictions, 1)

        return predicted.item() == True

class_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', 'Alaskan malamute', 'American eskimo dog',
               'American foxhound', 'American staffordshire terrier', 'American water spaniel',
               'Anatolian shepherd dog', 'Australian cattle dog', 'Australian shepherd', 'Australian terrier',
               'Basenji', 'Basset hound', 'Beagle', 'Bearded collie', 'Beauceron', 'Bedlington terrier',
               'Belgian malinois', 'Belgian sheepdog', 'Belgian tervuren', 'Bernese mountain dog', 'Bichon frise',
               'Black and tan coonhound', 'Black russian terrier', 'Bloodhound', 'Bluetick coonhound', 'Border collie',
               'Border terrier', 'Borzoi', 'Boston terrier', 'Bouvier des flandres', 'Boxer', 'Boykin spaniel',
               'Briard', 'Brittany', 'Brussels griffon', 'Bull terrier', 'Bulldog', 'Bullmastiff', 'Cairn terrier',
               'Canaan dog', 'Cane corso', 'Cardigan welsh corgi', 'Cavalier king charles spaniel',
               'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested', 'Chinese shar-pei', 'Chow chow',
               'Clumber spaniel', 'Cocker spaniel', 'Collie', 'Curly-coated retriever', 'Dachshund', 'Dalmatian',
               'Dandie dinmont terrier', 'Doberman pinscher', 'Dogue de bordeaux', 'English cocker spaniel',
               'English setter', 'English springer spaniel', 'English toy spaniel', 'Entlebucher mountain dog',
               'Field spaniel', 'Finnish spitz', 'Flat-coated retriever', 'French bulldog', 'German pinscher',
               'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer', 'Giant schnauzer',
               'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 'Great dane', 'Great pyrenees',
               'Greater swiss mountain dog', 'Greyhound', 'Havanese', 'Ibizan hound', 'Icelandic sheepdog',
               'Irish red and white setter', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound',
               'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier', 'Komondor', 'Kuvasz',
               'Labrador retriever', 'Lakeland terrier', 'Leonberger', 'Lhasa apso', 'Lowchen', 'Maltese',
               'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff', 'Newfoundland',
               'Norfolk terrier', 'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier',
               'Nova scotia duck tolling retriever', 'Old english sheepdog', 'Otterhound', 'Papillon',
               'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 'Petit basset griffon vendeen',
               'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese water dog', 'Saint bernard',
               'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff', 'Welsh springer spaniel',
               'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']


###############################
# DETECT BREED IF IT IS A DOG #
###############################


def predict_breed(model, img_path):
    image = Image.open(img_path).convert('RGB')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Declaring transforms and normalizing according to the documentation
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

    # discard the transparent, alpha channel and adding dimension (just as in the VGG prediction)
    image = transform(image)[:3, :, :].unsqueeze(0)

    model = model.to("cpu")  # changed to CPU because of it is expecting a floattensor, not a cuda.floattensor
    model.eval()
    idx = torch.argmax(model(image))  # the index with the highest prediction

    return class_names[idx]


model_transfer = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model_transfer.parameters():
    param.requires_grad = False

# Define new classifier, two fully connected layers with ReLU activation function and dropout
classifier = nn.Sequential(nn.Linear(512, 256),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(256, 133),
                           nn.LogSoftmax(dim=1))

model_transfer.fc = classifier

# check if CUDA is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    model_transfer = model_transfer.cuda()

model_transfer.load_state_dict(torch.load('./saved_model.pt', map_location=torch.device('cpu')))

########################################
# ADDITIONAL SUPPORT FOR DOGS AND CATS #
########################################

VGG16 = models.vgg16(pretrained=True)
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()


def VGG16_predict(img_path):
    image = Image.open(img_path).convert('RGB')

    # These are the transforms according to the documentation in pytorch.models (
    # https://pytorch.org/hub/pytorch_vision_vgg/)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # Apply transforms and add new dimension (to input in the network)
    image_tensor = transform(image).unsqueeze(0)

    # move the input and model to GPU
    if use_cuda:
        image_tensor = image_tensor.to('cuda')

    # We turn eval() mode on before prediction and resume training afterwards
    VGG16.eval()

    # Get prediction
    with torch.no_grad():
        output = VGG16(image_tensor)
        prediction = torch.argmax(output).item()  # Returns the index of the maximum value

    VGG16.train()

    return prediction


def dog_detector(img_path):
    prediction_index = VGG16_predict(img_path)
    return True if (269 > prediction_index > 150) else False


def cat_detector(img_path):
    prediction_index = VGG16_predict(img_path)
    return True if (288 > prediction_index > 280) else False

##################
# FINAL FUNCTION #
##################

def detect_image(img_path):
    ret_str = ""
    if face_detector(img_path) is True:
        ret_str += "There is a human in this photo.\n"

    if dog_or_cat_predict(img_path) is True:
        ret_str += "There is a high probability this has a dog...\n"
        if dog_detector(img_path) is True:
            pred = predict_breed(model_transfer, img_path)
            ret_str += "its breed being {0}.\n".format(pred)

    elif cat_detector(img_path) is True:
        ret_str += "I'm quite certain the picture has a cat!\n"

    else:
        ret_str += "There is not a human nor there is a dog, so perhaps a cat...?\n"

    return ret_str

