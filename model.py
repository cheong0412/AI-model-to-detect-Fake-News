# 6.19 양방향 다층 장단기 메모리
import torch
from torch import nn


input_size = 128
output_size = 256
num_layers = 3
bidirectional = True
proj_size = 64

model = nn.LSTM(
    input_size=input_size,
    hidden_size=output_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    proj_size=proj_size,
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size)
h_0 = torch.rand(
    num_layers * (int(bidirectional) + 1),
    batch_size,
    proj_size if proj_size > 0 else output_size,
)
c_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, output_size)

outputs, (h_n, c_n) = model(inputs, (h_0, c_0))

print(outputs.shape)
print(h_n.shape)
print(c_n.shape)

# -------------------------------------------------------------------------------------------------------

# 6.20 문장 분류 모델
from torch import nn


class SentenceClassifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        embedding_dim,
        n_layers,
        dropout=0.6,
        bidirectional=True,
        model_type="lstm"
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        if model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )

        if bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits

# -------------------------------------------------------------------------------------------------------

# df_train = pd.read_csv('./fakenews_preprocessed_trian2_.csv')
# test_df = pd.read_csv('./fakenews_preprocessed_test2_.csv')

# train = df_train.sample(frac=0.9, random_state=42)
# test = test_df


import pandas as pd

train = pd.read_csv('./fakenews_preprocessed_trian2_.csv').sample(frac=0.9, random_state=42)
test = pd.read_csv('./fakenews_preprocessed_test2_.csv')


# -------------------------------------------------------------------------------------------------------

# 6.22 데이터 토큰화 및 단어 사전 구축
import pickle
import ast
from collections import Counter

def build_vocab(corpus, n_vocab, special_tokens, save_path=None, save_path_txt=None):
    counter = Counter()
    for i, tokens in enumerate(corpus):
        counter.update(tokens)
        if (i + 1) % 10000 == 0:
            print(f"{i + 1}개의 문장을 처리했습니다.")
    vocab = special_tokens[:]
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocab 저장 완료 (pickle): {save_path}")

    if save_path_txt:
        with open(save_path_txt, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(f"{token}\n")
        print(f"Vocab 저장 완료 (txt): {save_path_txt}")

    print(f"최종 vocab 크기: {len(vocab)}개")
    return vocab


# content 컬럼의 문자열을 진짜 리스트로 변환
train_tokens = [ast.literal_eval(line) for line in train['content']]
test_tokens = [ast.literal_eval(line) for line in test['content']]

#vocab = build_vocab(corpus=train_tokens, n_vocab=80000, special_tokens=["<PAD>", "<UNK>"], save_path='vocab2.pkl', save_path_txt='vocab2.txt')

# print(vocab[:10])
# print(len(vocab))


# #
import os

VOCAB_PATH = "vocab2.pkl"

if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
else:
    vocab = build_vocab(
        corpus=train_tokens,
        n_vocab=80000,
        special_tokens=["<PAD>", "<UNK>"],
        save_path=VOCAB_PATH,
        save_path_txt="vocab.txt"
    )

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

# -------------------------------------------------------------------------------------------------------

# 6.23 정수 인코딩 및 패딩
import numpy as np


def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)


unk_id = token_to_id["<UNK>"]
train_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
]
test_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
]

max_length = 32
pad_id = token_to_id["<PAD>"]
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)

print(train_ids[0])
print(test_ids[0])


# -------------------------------------------------------------------------------------------------------

# 6.24 데이터로더 적용
import torch
from torch.utils.data import TensorDataset, DataLoader


train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

train_labels = torch.tensor(train.label.values, dtype=torch.float64)
test_labels = torch.tensor(test.label.values, dtype=torch.float64)

train_dataset = TensorDataset(train_ids, train_labels)
test_dataset = TensorDataset(test_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

# -------------------------------------------------------------------------------------------------------

# 6.25 손실 함수와 최적화 함수 정의
from torch import optim

n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = SentenceClassifier(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers
).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay=1e-5  -> 과적합 방지를 위해 파라미터 크기를 제한하는 코드 추가함.

# -------------------------------------------------------------------------------------------------------

# 6.26 모델 학습 및 테스트 -> pt 저장 (에포크 약 10)
def train(model, datasets, criterion, optimizer, device, interval=100):
    model.train()
    losses = list()
    
    corrects = 0  # 정확도 계산을 위한 초기화
    total = 0     # 총 샘플 개수를 계산하기 위한 초기화

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 예측값 구하기 -> 이진 분류라서 sigmoid 사용
        yhat = torch.sigmoid(logits) > 0.5
        # 정확도 계산
        corrects += torch.sum(yhat == labels).item()
        total += labels.size(0)

        if step % interval == 0:
            accuracy = corrects / total  # 정확도 계산
            print(f"Step {step}, Train Loss: {np.mean(losses)}, Train Accuracy: {accuracy * 100:.2f}%")

    print(f"Finished epoch with total {len(datasets)} samples.")
    return model


def test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        yhat = torch.sigmoid(logits) > 0.5
        corrects.extend(torch.eq(yhat, labels).cpu().tolist())

    val_loss = np.mean(losses)
    val_accuracy = np.mean(corrects)
    print(f"Val Loss : {val_loss}, Val Accuracy : {val_accuracy}")
    return val_accuracy


######## 에포크 반복 및 모델 저장 ########
epochs = 10
interval = 100

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    classifier.train()
    losses = []
    corrects = 0
    total = 0
    processed_samples = 0

    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = classifier(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 예측값 구하기
        yhat = torch.sigmoid(logits) > 0.5
        
        # 정확도 계산
        corrects += torch.sum(yhat == labels).item()
        total += labels.size(0)

        # 배치마다 처리된 데이터 수 추적
        processed_samples += labels.size(0)

        if step % interval == 0:
            accuracy = corrects / total  # 정확도 계산
            print(f"Step {step}, Train Loss: {np.mean(losses)}, Train Accuracy: {accuracy * 100:.2f}%")

    # 에포크 끝날 때 처리된 총 데이터 수 출력
    print(f"Epoch {epoch+1} finished. Total samples processed: {processed_samples}/{len(train_loader.dataset)}")

    # 모델 테스트 후 정확도 계산
    accuracy = test(classifier, test_loader, criterion, device)

    # 모델 저장 (.pt 파일, 파일명에 에포크 번호와 정확도 포함)
    save_path = f"model_epoch_{epoch+1}_accuracy_{accuracy:.4f}.pt"
    torch.save(classifier.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# -------------------------------------------------------------------------------------------------------
