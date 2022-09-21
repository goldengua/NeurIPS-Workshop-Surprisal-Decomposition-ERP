##!pip install transformers
##!pip install levenshtein

import math
from collections import defaultdict
import Levenshtein
import torch
import pandas as pd
import numpy as np

from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base',top_k=100)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def editDistDP(str1,str2, cost, vowel_penalty, lbd):
  return math.exp(-Levenshtein.distance(str1, str2)*lbd)

def posterior(mask_sentence,true_sentence,cost,vowel_penalty,lbd):
  dic = unmasker(mask_sentence)
  post_dic = dic[:]
  for i in range(len(dic)):
    post_prob = dic[i]['score']*editDistDP(true_sentence,dic[i]['sequence'],cost,vowel_penalty,lbd)
    post_dic[i]['posterior_score'] = post_prob
  post_dic.sort(key=lambda x:x['posterior_score'],reverse=True)
  prob_sum = sum([post_dic[i]['posterior_score'] for i in range(len(post_dic))])
  if prob_sum != 0:
    for i in range(len(post_dic)):
      post_dic[i]['posterior_score'] = post_dic[i]['posterior_score']/prob_sum
  else:
    post_dic = [{'posterior_score':1,'token_str':' '+true_sentence.split()[-1].replace('.',''),'sequence':mask_sentence.replace(' <mask>.','')}]
  return post_dic

#get GPT2 probability
def prediction(text, true_target, post_dic):
    
    tokenized_text = tokenizer.tokenize(text)
    
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        predictions = model(tokens_tensor)
    prob_all = torch.softmax(predictions[0][0,-1,:],dim=0)

    tokenized_true_target = tokenizer.tokenize(true_target)
    true_target_index =  tokenizer.convert_tokens_to_ids(tokenized_true_target)
    prob = 0
    for idx in true_target_index:
      prob = prob + prob_all[idx].item()
    if len(true_target_index) != 0:
      prob = prob/len(true_target_index)
    true_surprisal = -math.log(prob)

    weighted_prob = 0
    for i in range(len(post_dic)):
      target = post_dic[i]['token_str']+'.'
      tokenized_target = tokenizer.tokenize(target)
      target_index =  tokenizer.convert_tokens_to_ids(tokenized_target)
      prob = 0
      for idx in target_index:
        prob = prob + prob_all[idx].item()
      if len(target_index) != 0:
        prob = prob/len(target_index)
      weighted_prob = weighted_prob + prob*post_dic[i]['posterior_score']
    heuristic_surprisal = -math.log(weighted_prob)
    structural_update = true_surprisal - heuristic_surprisal

    return true_surprisal, heuristic_surprisal, structural_update


def main(file,cost, vowel_penalty,lbd):
  df = pd.read_csv(file)
  true_surprisal_list, heuristic_surprisal_list, structural_update_list = [],[],[]
  for i in range(len(df)):
    #target = ' '+ df['sentence'][i].split()[-1].replace('.','')
    target = ' '+ df['sentence'][i].split()[-1]
    context = df['sentence'][i].replace(target,'')
    #context = df['sentence'][i].replace(target+'.','')
    masked_sentence = df['sentence'][i].replace(target.replace('.',''),' <mask>')
    #masked_sentence = df['sentence'][i].replace(target,'<mask>')
    true_sentence = df['sentence'][i]
    #print(target,context,masked_sentence,true_sentence)
    post_dic = posterior(masked_sentence,true_sentence,cost,vowel_penalty,lbd)
    true_surprisal, heuristic_surprisal, structural_update = prediction(context,target,post_dic)
    true_surprisal_list.append(true_surprisal)
    heuristic_surprisal_list.append(heuristic_surprisal)
    structural_update_list.append(structural_update)
    #print(i)
  df['true_surprisal'] = true_surprisal_list
  df['heuristic_surprisal'] = heuristic_surprisal_list
  df['structural_update'] = structural_update_list
  df.to_csv(f'result_roberta_base_{str(lbd)}_{file}')
  return df

###merge dataset
def merge_df(lbd):
  df = pd.read_csv(f'result_roberta_base_{lbd}_ryskin_stimuli.csv')
  df_sub = df[['item','condition','true_surprisal','heuristic_surprisal','structural_update']]
  df_human = pd.read_csv('ERP_subset.csv')
  df_human['item'] = df_human['itemNum']
  df_human_sub = df_human[['item','condition','subject','meanAmp','artefact','electrode','time_window']]
  df_merge = df_human_sub.merge(df_sub,on=['item','condition'])
  df_merge.to_csv(f'result_roberta_base_{lbd}_human_merged.csv')
  return df_merge