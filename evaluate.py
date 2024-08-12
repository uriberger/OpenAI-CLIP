import config as CFG
from CLIP import CLIPModel
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from main import build_loaders
from transformers import DistilBertTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
args = parser.parse_args()

model = CLIPModel().to(CFG.device)
model.load_state_dict(torch.load(args.checkpoint, map_location=CFG.device))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
test_loader = build_loaders(tokenizer, mode="test", batch_size=1)

image_embeddings = []
text_embeddings = []
image_path_to_ind = {}
image_ind_to_caption_inds = {}
caption_ind_to_image_ind = {}
with torch.no_grad():
    for batch in tqdm(test_loader, 'Computing embeddings'):
        image_path = batch['image_path'][0]
        if not image_path in image_path_to_ind:
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            cur_image_embeddings = model.image_projection(image_features)
            image_path_to_ind[image_path] = len(image_embeddings)
            image_ind_to_caption_inds[len(image_embeddings)] = []
            image_embeddings.append(cur_image_embeddings)
        image_ind = image_path_to_ind[image_path]
        text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))
        cur_text_embeddings = model.text_projection(text_features)
        image_ind_to_caption_inds[image_ind].append(len(text_embeddings))
        caption_ind_to_image_ind[len(text_embeddings)] = image_ind
        text_embeddings.append(cur_text_embeddings)

image_embedding_tensor = torch.cat(image_embeddings)
text_embedding_tensor = torch.cat(text_embeddings)

image_embeddings_n = F.normalize(image_embedding_tensor, p=2, dim=-1)
text_embeddings_n = F.normalize(text_embedding_tensor, p=2, dim=-1)
dot_similarity = text_embeddings_n @ image_embeddings_n.T
k_vals = [1, 5, 10]
_, indices = torch.topk(dot_similarity.squeeze(0), max(k_vals))

sample_num = len(caption_ind_to_image_ind)
res = {}
for k in k_vals:
    res[k] = len([i for i in range(sample_num) if caption_ind_to_image_ind[i] in indices[i][:k]])/sample_num

print(res)
