
from model import VQA_model
from preprocess import VQAdataset
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import torchvision.transforms as transforms


def evaluate_hw2():
    transform = transforms.Compose([transforms.ToTensor()
                            ,transforms.Normalize((0.4695836035981774, 0.44679106984101236, 0.40951072280388323)
                                                 ,(0.24364206719957293, 0.2388205561041832, 0.24255008994787933))])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = pickle.load( open( "vqa_val_dataset.pkl", "rb" ))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    word_emb_dim = 75
    vocab_len = len(val_dataset.word_dict)
    lstm_hidden_dim = 100
    region_emb_dim = 150
    num_regions = 7*7
    self_interaction_dim = 75
    interaction_dim = 100
    n_answers = 1276

    model = VQA_model(word_emb_dim,num_regions,  vocab_len,lstm_hidden_dim, region_emb_dim, self_interaction_dim,
                      interaction_dim, n_answers)
    model.load_state_dict(torch.load('VQA_model_epoch:15.pkl', map_location=lambda storage, loc: storage))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()


    with torch.no_grad():
        correct = 0
        for (image_filenames, questions, answers) in val_dataloader:
            batch_size = len(image_filenames)
            for i, image_filename in enumerate(image_filenames):
                if i == 0:
                    images = transform(pickle.load(open(f"./images/{image_filename}", "rb")))
                    continue
                images = torch.cat((images, transform(pickle.load(open(f"./images/{image_filename}", "rb")))))
            images = images.view((len(image_filenames), 3, 224, 224)).to(device)
            questions = questions.to(device)
            loss, predictions = model(images, questions, answers)
            for labels, scores in zip(answers["labels"], answers["scores"]):
                for j in range(batch_size):
                    if predictions[j] == labels[j]:
                        correct += float(scores[j])

    print(f"val accuracy:{correct/len(val_dataset)}")

if __name__ == '__main__':
    evaluate_hw2()