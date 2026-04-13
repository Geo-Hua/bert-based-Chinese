import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import re
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def judge_emotion(input_file):
    # 读取CSV文件
    df_train = pd.read_csv('virus_train1.csv')

    comments = df_train['comment'].tolist()
    labels = df_train['label'].tolist()


    # Label encoding: converting emotional labels to numbers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # 将情感标签转为数字

    # Initialising the BERT model and tokenizer
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)  # 7种情感类别

    # Define the dataset class
    class SentimentDataset(Dataset):
        def __init__(self, comments, labels, tokenizer, max_len=128):
            self.comments = comments
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.comments)

        def __getitem__(self, idx):
            comment = self.comments[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                comment,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


    # Divide the training set and validation set
    train_comments, val_comments, train_labels, val_labels = train_test_split(comments, encoded_labels, test_size=0.1)

    # Creating a Data Loader
    train_dataset = SentimentDataset(train_comments, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_comments, val_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} complete. Loss: {loss.item()}")

    # save modelling
    model.save_pretrained("sentiment_model")
    tokenizer.save_pretrained("sentiment_model")

    # Prediction of new comments using a trained model
    def predict_sentiment(comments):
        model.eval()
        predictions = []
        with torch.no_grad():
            for comment in comments:
                encoding = tokenizer.encode_plus(
                    comment,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # 使用softmax得到每种情感的概率
                predictions.append(probs.squeeze().cpu().numpy())

        return predictions


    # Loading the emoji dictionary
    with open('../data/emoji_Chinese.json', 'r', encoding='utf-8') as f:
        emoji_list = json.load(f)

    # Converting lists to dictionary form
    emoji_dict = {list(item.keys())[0]: list(item.values())[0] for item in emoji_list}


    # Replace the emoji in the comment with the corresponding text
    def replace_emoji_with_text(text):
        for emoji, meaning in emoji_dict.items():
            text = text.replace(emoji, meaning)
        return text
    # Calculate the ratio of positive to negative emotions
    def calculate_sentiment_ratio(emotion_dict):
        positive_emotions = ['like', 'surprise', 'happy']  # 假设这三个是积极情绪
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']  # 这些是消极情绪

        positive_score = sum(emotion_dict[emotion] for emotion in positive_emotions)
        negative_score = sum(emotion_dict[emotion] for emotion in negative_emotions)

        total_score = positive_score + negative_score
        if total_score > 0:
            positive_ratio = positive_score / total_score
            negative_ratio = negative_score / total_score
        else:
            positive_ratio = 0
            negative_ratio = 0

        return positive_ratio, negative_ratio


    df_test = pd.read_csv(input_file)
    df_test['content'] = df_test['content'].apply(replace_emoji_with_text)
    test_comments = df_test['content'].tolist()

    review = df_test.copy()
    # prediction
    probs = predict_sentiment(test_comments)

    results = []
    pos_neg=[]
    for idx, prob in enumerate(probs):
        emotion_dict = {label: p for label, p in zip(label_encoder.classes_, prob)}
        total_prob = prob.sum()  # Ensure that the sum is 1

        # Calculate the positive and negative ratio of emotions
        positive_ratio, negative_ratio = calculate_sentiment_ratio(emotion_dict)
        pos_neg_dict = {'positive':positive_ratio,'negative':negative_ratio}
        # save result
        result = emotion_dict.copy()
        results.append(result)
        pos_neg.append(pos_neg_dict)

    review['sentiment']=results
    review['pos-neg'] = pos_neg

    results_df = pd.DataFrame(review)
    results_df.to_csv('../data/emotion_prediction_wh.csv', index=False,encoding='utf-8-sig')
    print("情感分析结果已保存到 'emotion_predictions.csv'")


    def calculate_metrics(predictions, true_labels):
        # 将预测的概率最大值索引作为预测标签
        predicted_labels = np.argmax(predictions, axis=1)

        # calculate Accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculate Precision, Recall, F1 for each category
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

        # Calculate the weighted average of Precision, Recall, F1
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')


        overall_metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted
        }

        per_class_metrics = {
            "precision": precision_per_class,
            "recall": recall_per_class,
            "f1": f1_per_class
        }

        return overall_metrics, per_class_metrics


    def evaluate_model(model, val_dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Getting the output of the model
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)  # 得到每种情感的概率

                # Converting predicted probabilities to labels
                preds = probs.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculation of indicators
        overall_metrics, per_class_metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))

        return overall_metrics, per_class_metrics

    # Evaluation of the validation set after training
    overall_metrics, per_class_metrics = evaluate_model(model, val_dataloader, device)

    # Export of corporate evaluation indicators
    print(f"Validation Accuracy: {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Precision (Weighted): {overall_metrics['precision_weighted'] * 100:.2f}%")
    print(f"Validation Recall (Weighted): {overall_metrics['recall_weighted'] * 100:.2f}%")
    print(f"Validation F1 Score (Weighted): {overall_metrics['f1_weighted'] * 100:.2f}%")

    # Output Precision, Recall, F1 for each emotion category
    print("\nPer-Class Metrics:")
    for i, (p, r, f) in enumerate(zip(per_class_metrics['precision'], per_class_metrics['recall'], per_class_metrics['f1'])):
        print(f"Class {i} - Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}")

if __name__=="__main__":
    judge_emotion('../data/wh_data_cleaned.csv')