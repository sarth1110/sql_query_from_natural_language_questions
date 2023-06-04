
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import copy
import gzip
import json
import re
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readLines(path):
  fptr = open(path, 'r')
  lines = fptr.readlines()
  return lines


trainQuery = readLines('data/data/train.jsonl')
devQuery = readLines('data/data/dev.jsonl')
trainTables = readLines('data/data/train.tables.jsonl')
devTables = readLines('data/data/dev.tables.jsonl')

def prepareQueryData(queryData):
    tabId = []
    question = []
    SQL = []
    # selectCol = []
    whereCondition = []
    # aggOperator = []
    whereCol = []
    whereOp = []
    whereVal = []
    for i, query in enumerate(queryData):
        q = json.loads(query)
        if len(q["sql"]["conds"]) > 0:
            tabId.append(q["table_id"])
            question.append(q["question"])
            SQL.append(q["sql"])
            whereCondition.append(q["sql"]["conds"])
            whereCol.append(q["sql"]["conds"][0][0])
            whereOp.append(q["sql"]["conds"][0][1])
            whereVal.append(q["sql"]["conds"][0][2])
    return tabId, question, SQL, whereCondition, whereCol, whereOp, whereVal

def prepareTableData(tableData, tableID, whereCol):
    columnName = []
    columnDict = {}
    allColumnList = []
    targetWhereColumn = []

    for i, data in enumerate(tableData):
        q = json.loads(data)
        columnDict[q['id']] = q['header']

    for id in tableID:
        columnName.append(columnDict[id])
        allColumnList.extend(columnDict[id])

    for i, col in enumerate(whereCol):
        targetWhereColumn.append(columnName[i][col])
    
    return columnName, allColumnList, targetWhereColumn

def remove_puctuations(temp):
    text = temp
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    if text==" " or text=="":
        return temp
    return text

def updateMapping(globalColumns, vocabMapping):
    count = len(vocabMapping)
    for col in globalColumns:
      for i in col.split():
        if(i.lower() not in vocabMapping):
            vocabMapping[i.lower()] = count
            count+=1
    return vocabMapping


def prepareData(lines):
    sentences = []
    for line in lines:
        lst = line.split(' ')
        sentences.append(lst)
    return sentences


def createMapping(sentences):
    vocabMapping = {}
    vocabMapping['<pad>'] = 0
    vocabMapping['<unk>'] = 1
    i=2
    vocab = []
    for words in sentences:
        vocab.extend(words)

    vocabDict = {}
    for v in vocab:
        if v.lower() in vocabDict:
            vocabDict[v.lower()]+=1
        else:
            vocabDict[v.lower()]=1

    vocab = list(vocabDict.keys())
    for word in vocab:
        vocabMapping[word.lower()] = i
        i+=1
    return vocabMapping

def prepareSentences(lines, vocabMapping):
    sentences = []

    for line in lines:
      sentence = []

      for word in line:
        if(word.lower() in vocabMapping):
            sentence.append(vocabMapping[word.lower()])
        else:
            sentence.append(vocabMapping['<unk>'])
      sentences.append(sentence)

    sentenceLengths = [len(x) for x in sentences]
    maxLen = max(sentenceLengths)
    listS = []

    for sentence in sentences:
        padLength = maxLen - len(sentence)
        padLength = max(0, padLength)
        sentence.extend([0 for i in range(padLength)])
        # print(sentence)
        listS.append(sentence)

    return listS, sentenceLengths



def createSelectionData(columns, vocabMapping):
    colsAfterPad = []
    finalColLength = []
    colsBeforePad = []
    columns = [remove_puctuations(x) for x in columns]
    for col in columns:
        temp = []
        for words in col.split():
            if(words.lower() in vocabMapping):
                temp.append(vocabMapping[words.lower()])
            else:
                temp.append(vocabMapping['<unk>'])
        colsBeforePad.append(temp)

    colLengths = [len(x) for x in colsBeforePad]
    maxLen = max(colLengths)

    for tableCols in (colsBeforePad):
        padLength = maxLen - len(tableCols)
        padLength = max(0, padLength)
        colsAfterPad.append(tableCols+[0 for i in range(padLength)])
        
    return colsAfterPad, colLengths


def createDataLoader(question, questionLength, column, columnLength, targetColumn):
    dataset = TensorDataset(torch.Tensor([question]), torch.Tensor([questionLength]), torch.Tensor([column]) ,torch.Tensor([columnLength]) ,torch.Tensor([targetColumn]))
    dataLoader = DataLoader(dataset, batch_size= 1, shuffle=True)
    return dataLoader

def createPreTrainedEmbed(vocabMapping):
    gloveEmbeddings = {}
    with open('glove.twitter.27B.50d.txt', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:],dtype='float32')
            gloveEmbeddings[word] = embedding

    wordEmbedding = np.zeros((len(vocabMapping), 50))
    for word in vocabMapping:
        if(word.lower() in gloveEmbeddings):
            wordEmbedding[vocabMapping[word]] = gloveEmbeddings[word.lower()]
        else:
            wordEmbedding[vocabMapping[word]] = gloveEmbeddings['unk']
    wordEmbeds = torch.Tensor(wordEmbedding)
    return wordEmbeds


trainTabId, trainQuestion, trainSQL, trainCon, trainWhereCol, trainWhereOp, trainWhereVal = prepareQueryData(trainQuery)
devTabId, devQuestion, devSQL, devCon, devWhereCol, devWhereOp, devWhereVal = prepareQueryData(devQuery)

trainColumns, globalColumns, trainWhereTarget = prepareTableData(trainTables, trainTabId, trainWhereCol)
devColumns, _, devWhereTarget = prepareTableData(devTables, devTabId, devWhereCol)


trainQuestion = [remove_puctuations(x) for x in trainQuestion]
devQuestion = [remove_puctuations(x) for x in devQuestion]
trainQuestions = prepareData(trainQuestion)
devQuestions = prepareData(devQuestion)
vocabMapping = createMapping(trainQuestions)
trainQuestions, questionLength = prepareSentences(trainQuestions, vocabMapping)
devQuestions, devLength = prepareSentences(devQuestions, vocabMapping)

globalColumns = [remove_puctuations(x) for x in globalColumns]
vocabMapping = updateMapping(globalColumns, vocabMapping)
wordEmbeds = createPreTrainedEmbed(vocabMapping)

class BiLSTM(nn.Module):
    def __init__(self, embeddingDim=50, lstmHiddenDim=128, linearOutDim=128, vocabSize=0, tagSize=0, pretrainedEmbed=None, freeze=True):
        super(BiLSTM, self).__init__()        
        self.wordembed = nn.Embedding.from_pretrained(pretrainedEmbed, freeze = freeze)
        self.bilstm1 = nn.LSTM(input_size = embeddingDim, hidden_size = lstmHiddenDim, bidirectional = True, batch_first = True, num_layers=1)
        self.bilstm2 = nn.LSTM(input_size = embeddingDim, hidden_size = lstmHiddenDim, bidirectional = True, batch_first = True, num_layers=1)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(2*lstmHiddenDim, 2*linearOutDim, bias=False)
        self.linear2 = nn.Linear(2*lstmHiddenDim, 2*linearOutDim, bias=False)
        self.linear3 = nn.Linear(2*linearOutDim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.linearL1 = nn.Linear(2*lstmHiddenDim, 2*lstmHiddenDim)
        self.linearL2 = nn.Linear(2*lstmHiddenDim, 2*lstmHiddenDim)
        self.linearPsel = nn.Linear(2*lstmHiddenDim, 1)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, que, queLength, cols, colLength):
        que = pack_padded_sequence(que, queLength.cpu(), batch_first=True, enforce_sorted=False)
        que, _ = pad_packed_sequence(que, batch_first=True)
        wordEmbeddingQue = self.wordembed(que)
        outQue, _ = self.bilstm1(wordEmbeddingQue)
        colsShape = cols.shape
        cols=torch.reshape(cols, (colsShape[1], colsShape[2]))
        cols = pack_padded_sequence(cols, colLength[0].cpu(), batch_first=True, enforce_sorted=False)
        cols, _ = pad_packed_sequence(cols, batch_first=True)
        wordEmbeddingCol = self.wordembed(cols)
        outCol, _ = self.bilstm2(wordEmbeddingCol)
        outCol = outCol[:,-1,:]
        outColL = self.linear1(outCol)
        outColL = torch.transpose(outColL, 0, 1)
        alpha = torch.matmul(outQue, outColL)
        alpha = self.softmax1(alpha)
        
        outQueCol = torch.matmul(torch.transpose(alpha, 1, 2), outQue)
        L1 = self.linearL1(outCol)
        L2 = self.linearL2(outQueCol[0])
        Psel = self.linearPsel(self.tanh(torch.add(L1, L2)))
        Psel = self.sigmoid(Psel)
        return Psel

model = BiLSTM(pretrainedEmbed=wordEmbeds, freeze=False)
model = model.to(device)

def trainModel(model, epoch, trainQuestions, questionLength, trainColumns, trainWhereCol):
    num_epochs = epoch
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    lossfunction = nn.CrossEntropyLoss()
    best_model = None
    best_epoch = None
    best_f1 = -1
    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i in range(len(trainQuestions)):
            trainColsAfterPad, trainColLengths = createSelectionData(trainColumns[i], vocabMapping)
            trainLoader = createDataLoader(trainQuestions[i], questionLength[i], trainColsAfterPad, trainColLengths, trainWhereCol[i])
            optimizer.zero_grad()
            for j, (question, queLength, column, columnLength, targetColumn) in enumerate(trainLoader):
                question = question.to(device)
                queLength = queLength.to(device)
                column = column.to(device)
                columnLength = columnLength.to(device)
                targetColumn = targetColumn.to(device)
                ypred = model(question.long(), queLength, column.long(), columnLength)
                ypred = torch.reshape(ypred, (ypred.shape[0], ))

                loss = lossfunction(torch.stack([ypred]).to(device), targetColumn.long().to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss = train_loss/len(trainQuestions)
        print('Epoch: {}\t --->\tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    return model


model = trainModel(model, 20, trainQuestions, questionLength, trainColumns, trainWhereCol)

torch.save(model.state_dict(), "where_col.pt")

def evalModel(model, devQuestions, devLength, devColumns, devWhereCol):
    model.eval()
    pred = []
    true = [] 
    with torch.no_grad():
      for i in range(len(devQuestions)):
            devColsAfterPad, devColLengths = createSelectionData(devColumns[i], vocabMapping)
            devLoader = createDataLoader(devQuestions[i], devLength[i], devColsAfterPad, devColLengths, devWhereCol[i])
            for j, (question, questionLength, column, columnLength, targetColumn) in enumerate(devLoader):
                question = question.to(device)
                questionLength = questionLength.to(device)
                column = column.to(device)
                columnLength = columnLength.to(device)
                targetColumn = targetColumn.to(device)
                ypred = model(question.long(), questionLength, column.long(), columnLength)
                pred.append(torch.argmax(ypred, dim=0).float().tolist())
                true.append(targetColumn[0].tolist())
    return f1_score(true, pred, average="macro"), accuracy_score(true, pred)

evalModel(model, devQuestions, devLength, devColumns, devWhereCol)




