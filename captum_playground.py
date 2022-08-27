# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import json
import torch
import argparse
import operator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from transformers import RobertaTokenizerFast
from captum.attr import (
    LayerIntegratedGradients,
    LayerDeepLift,
    LayerGradientXActivation,
    LayerFeatureAblation,
    TokenReferenceBase
)
from models import RobertaMLP
from utils import max_sublists
from constants import ID2LABEL, LABEL2ID


methods_register = {
    'LayerIntergratedGradients': ['lig', 'LayerIntergratedGradients', 'Layer Intergrated Gradients'],
    'LayerDeepLift': ['ldl', 'LayerDeepLift', 'Layer Deep Lift'],
    'LayerGradientXActivation': ['lgxa', 'LayerGradientXActivation', 'Layer Gradient X Activation'],
    'LayerFeatureAblation': ['lfa', 'LayerFeatureAblation', 'Layer Feature Ablation']
}


class CaptumInterpreter(object):
    def __init__(
        self, 
        method, 
        model_path, 
        pretrained_name_or_path, 
        tokenizer_path, 
        device,
        **kwargs
    ) -> None:
        self.method = method
        
        self.model = RobertaMLP(bert_path=pretrained_name_or_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.tokenizer = RobertaTokenizerFast(
            f"{tokenizer_path}/vocab.json",
            f"{tokenizer_path}/merges.txt",
        )
        self.token_reference = TokenReferenceBase(reference_token_idx=self.tokenizer.pad_token_id)

        if method in methods_register['LayerIntergratedGradients']:
            self.interpreter = LayerIntegratedGradients(self.model, self.model.model.embeddings)
        elif method in methods_register['LayerDeepLift']:
            self.interpreter = LayerDeepLift(self.model, self.model.model.embeddings)
        elif method in methods_register['LayerGradientXActivation']:
            self.interpreter = LayerGradientXActivation(self.model, self.model.model.embeddings)
        elif method in methods_register['LayerFeatureAblation']:
            self.interpreter = LayerFeatureAblation(self.model, self.model.model.embeddings)
        else:
            raise ValueError(
                f"The `method`=`{method}` dont supported !"
                f"Please select among them: ['lig', 'ldl', 'lgxa', 'lfa']"
            )

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def interpret_sample(self, text, max_length: int=200, label: int=0, target: int=0):
        self.model.zero_grad()
        self.model.to(self.device)

        input = self.tokenizer(text, max_length=max_length, truncation='only_first')
        input_indices = torch.tensor(input['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(input['attention_mask']).unsqueeze(0).to(self.device)

        #TODO: predict
        pred = self.model(input_indices, attention_mask)
        pred_prob = nn.functional.softmax(pred, dim=-1)
        pred = torch.argmax(pred_prob, dim=-1).item()
        pred_prob = pred_prob[0][pred].item()

        #TODO: generate reference indices for each sample
        reference_indices = self.token_reference.generate_reference(
            len(input_indices[0]), 
            device=self.device).unsqueeze(0)

        #TODO: compute attributions and approximation delta using layer integrated gradients
        attribution, delta = self.interpreter.attribute(
            inputs=input_indices,
            base_lines=reference_indices,
            target=target,
            additional_forward_args=attention_mask,
            n_steps=100,
            return_convergence_delta=True,
        )

        attribution = attribution.sum(dim=2).squeeze(0)
        attribution = attribution / torch.norm(attribution)
        attribution = attribution.cpu().detach().numpy()

        return list(attribution), delta

    def get_label(self, instance):
        tags = instance['tags']

        if not tags:
            return 'NEGATIVE'

        count_dict = {
            'NEGATIVE': 0,
            'POSITIVE': 0,
            'NEUTRAL': 0
        }
        opinion_terms = []
        for tag in tags:
            if tag['polarity'] == 'NEUTRAL':
                count_dict['POSITIVE'] += 1
            else:
                count_dict[tag['polarity']] += 1

            opinion_terms.append(tag['target'])

        returned_value = max(count_dict.items(), key=operator.itemgetter(1))[0]

        return returned_value, opinion_terms

    def interpret_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        attributions, deltas = [], []
        texts, terms = [], []

        sentences = data['document']['sentences']
        for sent in tqdm(sentences, desc="Processing"):
            label, term = self.get_label(sent)
            text = sent['content'].lower()
            term = [t.lower() for t in term]
            
            texts.append(text)
            terms.append(term)

            interpreted_sample = self.interpret_sample(
                text=text,
                max_length=200,
                label=label,
                target=1
            )
            attributions.append(interpreted_sample[0])
            deltas.append(interpreted_sample[1])

        return (attributions, deltas), texts, terms

    def get_text_from_interpreted(self, attributions, deltas, texts):
        

    def calculate_scores(self, attributions, deltas, texts, terms, threshold: float=0.4):
        scores = []





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='ldl', type=str,
                        help='Type of explanations algorithm')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Device')
    
    args = parser.parse_args()

