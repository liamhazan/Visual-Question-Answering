import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
from cnn_v2 import I_encoder
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class VQA_model(nn.Module):
    def __init__(self, word_emb_dim,num_regions, vocab_len,lstm_hidden_dim, region_emb_dim, self_interaction_dim,
                 interaction_dim, n_answers):
        super(VQA_model, self).__init__()


        self.q_len = 23
        self.region_emb_dim = region_emb_dim
        self.word_embedding = nn.Embedding(vocab_len, word_emb_dim, padding_idx=1)
        self.Q_encoder = nn.LSTM(word_emb_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.I_encoder = I_encoder()
        # self.I_encoder.load_state_dict(torch.load("epoch_7_self_cnn.pkl", map_location=torch.device(device)))

        self.I_entity_information = nn.Sequential(
            nn.Linear(region_emb_dim, region_emb_dim),
            nn.LeakyReLU(),
            nn.Linear(region_emb_dim, 1)
        )
        self.Q_entity_information = nn.Sequential(
            nn.Linear(lstm_hidden_dim*2, lstm_hidden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(lstm_hidden_dim*2, 1)
        )

        self.cos_sim = nn.CosineSimilarity(2)

        self.QI_L_interact = nn.Linear(lstm_hidden_dim * 2, interaction_dim)
        self.QI_R_interact = nn.Linear(region_emb_dim, interaction_dim)
        self.IQ_L_interact = nn.Linear(region_emb_dim, interaction_dim)
        self.IQ_R_interact = nn.Linear(lstm_hidden_dim * 2, interaction_dim)

        self.Q_L_self_interact = nn.Linear(lstm_hidden_dim*2, self_interaction_dim)
        self.Q_R_self_interact = nn.Linear(lstm_hidden_dim*2, self_interaction_dim)
        self.I_L_self_interact = nn.Linear(region_emb_dim, self_interaction_dim)
        self.I_R_self_interact = nn.Linear(region_emb_dim, self_interaction_dim)

        self.W_I = nn.Linear(num_regions, num_regions)
        self.W_II = nn.Linear(num_regions, num_regions)
        self.W_IQ = nn.Linear(num_regions, num_regions)

        self.W_Q = nn.Linear(self.q_len, self.q_len)
        self.W_QQ = nn.Linear(self.q_len, self.q_len)
        self.W_QI = nn.Linear(self.q_len, self.q_len)

        self.softmax = nn.Softmax(dim=1)

        self.ce_loss = nn.CrossEntropyLoss(weight=pickle.load( open( "class_weights.pkl", "rb" )))

        self.decision = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2 + region_emb_dim,n_answers),
            nn.LeakyReLU(),
            nn.Linear(n_answers,n_answers)
        )


    def forward(self, images, q_idxs, answers):

        q_len = 23
        q_embed = self.word_embedding(q_idxs)
        encoded_q, _ = self.Q_encoder(q_embed) #[BATCH_SIZE,q_len, lstm_hidden_dim * 2]
        batch_size = encoded_q.size(0)
        # t0 = time.time()
        encoded_regions = self.I_encoder(images).view((batch_size,-1,self.region_emb_dim)) #[BATCH_SIZE, num_regions, region_emb_dim]
        # t1 = time.time()
        num_regions = encoded_regions.shape[1]
        Q_entitiy_scores = self.Q_entity_information(encoded_q).view(batch_size,q_len) #[BATCH_SIZE, q_len]
        I_entitiy_scores = self.I_entity_information(encoded_regions).view(batch_size,num_regions) #[BATCH_SIZE, num_regions]
        # print(Q_entitiy_scores.shape)
        # print(I_entitiy_scores.shape)

        Q_self_interactions_scores = self.cos_sim(self.Q_L_self_interact(encoded_q).repeat(1, q_len, 1),
                                                  self.Q_R_self_interact(encoded_q).repeat(1, q_len, 1))\
                                                    .view(batch_size,q_len,q_len).sum(1) #[BATCH_SIZE, q_len]
        # print(Q_self_interactions_scores.shape)
        I_self_interactions_scores = self.cos_sim(self.I_L_self_interact(encoded_regions).repeat(1, num_regions, 1),
                                                  self.I_R_self_interact(encoded_regions).repeat(1, num_regions,1))\
                                                    .view(batch_size,num_regions,num_regions).sum(1) #[BATCH_SIZE, num_regions]
        # print(I_self_interactions_scores.shape)


        IQ_interaction_scores = self.cos_sim(self.IQ_L_interact(encoded_regions).repeat(1,q_len,1)
                                             ,self.IQ_R_interact(encoded_q).repeat(1,num_regions,1))\
                                                .view(batch_size, q_len, num_regions).sum(1) #[BATCH_SIZE, num_regions]
        # print(IQ_interaction_scores.shape)
        QI_interaction_scores = self.cos_sim(self.QI_L_interact(encoded_q).repeat(1, num_regions, 1)
                                             , self.QI_R_interact(encoded_regions).repeat(1, q_len, 1)) \
            .view(batch_size, num_regions, q_len).sum(1) #[BATCH_SIZE, q_len]
        # print(QI_interaction_scores.shape)



        b_I = self.softmax(self.W_I(I_entitiy_scores)+self.W_IQ(IQ_interaction_scores)+self.W_II(I_self_interactions_scores)).unsqueeze(1) #[BATCH_SIZE, 1, num_regions]
        b_Q = self.softmax(self.W_Q(Q_entitiy_scores)+self.W_QI(QI_interaction_scores)+self.W_QQ(Q_self_interactions_scores)).unsqueeze(1) #[BATCH_SIZE, 1, q_len]
        # print(b_I.shape)
        # print(b_Q.shape)
        alpha_I = torch.matmul(b_I,encoded_regions).squeeze(1) #[BATCH_SIZE, region_emb_dim]
        alpha_Q = torch.matmul(b_Q,encoded_q).squeeze(1) #[BATCH_SIZE, lstm_hidden_dim * 2]
        # print(alpha_I.shape)
        # print(alpha_Q.shape)
        prediction = self.decision(torch.cat([alpha_I,alpha_Q],dim=1))
        # print(prediction.shape)
        # t2 = time.time()



        loss = torch.tensor([0.0], requires_grad=True).to(device)
        total = torch.tensor([0]).to(device)
        prediction_ = self.softmax(prediction.clone())
        for i in range(batch_size):

            for labels, scores in zip(answers["labels"], answers["scores"]):
                labels = labels.to(device)
                scores = scores.to(device)
                curr_loss = self.ce_loss(prediction[i].unsqueeze(0), labels[i].unsqueeze(0))*(torch.pow(scores[i], exponent=2)) if scores[i] >= 0.3 else 0
                if curr_loss > 0 :
                    loss = torch.add(loss, curr_loss)
                    total += 1
        loss = torch.div(loss, total)
        # t3 = time.time()
        # print(f"cnn took {t1-t0} \n rest took {t2-t1} \n loss took {t3-t2}")
        return loss, torch.argmax(prediction_, dim=1)

# image = Image.open("./images/COCO_train2014_000000000025.jpg")
# print(image.size)
# t = transforms.Scale((300,300))
# t(image).show()

