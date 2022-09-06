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
    TokenReferenceBase,
    visualization
)
from models import RobertaMLP
from visualize import CaptumVisualizer
from constants import ID2LABEL, LABEL2ID
from utils import max_sublists, get_char_idx2token_idx

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
            is_visualize: bool=False,
            **kwargs
    ) -> None:
        self.method = method

        self.model = RobertaMLP(num_labels=2, bert_path=pretrained_name_or_path)
        self.model.mlp.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.tokenizer = RobertaTokenizerFast(
            f"{tokenizer_path}/vocab.json",
            f"{tokenizer_path}/merges.txt",
        )
        self.token_reference = TokenReferenceBase(reference_token_idx=self.tokenizer.pad_token_id)

        if method in methods_register['LayerIntergratedGradients']:
            self.interpreter = LayerIntegratedGradients(self.model, self.model.encoder.embeddings)
        elif method in methods_register['LayerDeepLift']:
            self.interpreter = LayerDeepLift(self.model, self.model.encoder.embeddings)
        elif method in methods_register['LayerGradientXActivation']:
            self.interpreter = LayerGradientXActivation(self.model, self.model.encoder.embeddings)
        elif method in methods_register['LayerFeatureAblation']:
            self.interpreter = LayerFeatureAblation(self.model, self.model.encoder.embeddings)
        else:
            raise ValueError(
                f"The `method`=`{method}` dont supported !"
                f"Please select among them: ['lig', 'ldl', 'lgxa', 'lfa']"
            )

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.captum_visualizer = None
        if is_visualize:
            self.visual_data_record = []
            self.captum_visualizer = CaptumVisualizer(tokenizer=self.tokenizer)

    def interpret_sample(self, text, max_length: int = 200, label: int = 0, target: int = 0):
        self.model.zero_grad()
        self.model.to(self.device)

        input = self.tokenizer(text, max_length=max_length, truncation='only_first')
        input_indices = torch.tensor(input['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(input['attention_mask']).unsqueeze(0).to(self.device)

        # TODO: predict
        pred = self.model(input_indices, attention_mask)
        pred_prob = nn.functional.softmax(pred, dim=-1)
        pred = torch.argmax(pred_prob, dim=-1).item()
        pred_prob = pred_prob[0][pred].item()
        rtext = input_indices

        # TODO: generate reference indices for each sample
        reference_indices = self.token_reference.generate_reference(
            len(input_indices[0]),
            device=self.device).unsqueeze(0)

        # TODO: compute attributions and approximation delta using layer integrated gradients
        if self.method in methods_register['LayerIntergratedGradients']:
            attribution, delta = self.interpreter.attribute(
                inputs=input_indices,
                baselines=reference_indices,
                target=target,
                additional_forward_args=attention_mask,
                n_steps=1000,
                return_convergence_delta=True,
            )
        else:
            attribution, delta = self.interpreter.attribute(
                inputs=input_indices,
                baselines=reference_indices,
                target=target,
                additional_forward_args=attention_mask,
                return_convergence_delta=True,
            )

        attribution = attribution.sum(dim=2).squeeze(0)
        attribution = attribution / torch.norm(attribution)
        attribution = attribution.cpu().detach().numpy()

        rattribution = []
        for i in range(len(attribution)):
            if input_indices[0][i] not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
                rattribution.append(attribution[i])

        if self.captum_visualizer:
            return attribution, delta, pred_prob, pred, rtext
        else:
            return rattribution, delta

    def get_label(self, instance):
        tags = instance['tags']

        if not tags:
            return 'NEGATIVE', []

        count_dict = {
            'NEGATIVE': 0,
            'POSITIVE': 0,
            'NEUTRAL': 0
        }
        opi_terms = []
        idx_opi_terms = []

        char_idx2token_idx = get_char_idx2token_idx(instance['content'].split())
        # print(char_idx2token_idx)

        for tag in tags:
            if tag['polarity'] == 'NEUTRAL':
                count_dict['POSITIVE'] += 1
            else:
                count_dict[tag['polarity']] += 1

            opi_terms.append(tag['target'])

            start_token_idx = char_idx2token_idx[tag['start_offset']]
            end_token_idx = char_idx2token_idx[tag['start_offset'] + len(tag['target']) - 1]
            idx_opi_terms.append((start_token_idx, end_token_idx))

        returned_value = max(count_dict.items(), key=operator.itemgetter(1))[0]

        return returned_value, idx_opi_terms, opi_terms

    def interpret_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        attributions, deltas = [], []
        texts, terms, idx_terms = [], [], []

        sentences = data['document']['sentences']
        for sent in tqdm(sentences, desc="Processing"):
            try:
                label, idx_term, term = self.get_label(sent)
            except:
                print(sent)
            text = sent['content'].lower()
            term = [t.lower() for t in term]
            texts.append(text)
            terms.append(term)
            idx_terms.append((idx_term))

            interpreted_sample = self.interpret_sample(
                text=text,
                max_length=200,
                label=label,
                target=LABEL2ID[label],
            )
            attributions.append(interpreted_sample[0])
            deltas.append(interpreted_sample[1])

        return (attributions, deltas), texts, (idx_terms, terms)

    def get_output_from_interpreted(self, attributions, deltas, texts, threshold: float = 0.4):
        text_spans = []
        idx_spans = []

        for i in range(len(attributions)):
            tmp_idx_span = []
            tmp_text_span = []
            pattern_found = []
            tmp_text = texts[i].split()
            for j in range(len(attributions[i])):
                if attributions[i][j] >= threshold:
                    pattern_found.append(j)
                elif len(pattern_found) != 0:
                    tmp_idx_span.append((pattern_found[0], pattern_found[-1]))
                    tmp_text_span.append(' '.join(tmp_text[pattern_found[0]: pattern_found[-1] + 1]))
                    pattern_found = []

            if len(pattern_found) != 0:
                tmp_idx_span.append((pattern_found[0], len(attributions[i]) - 1))
                tmp_text_span.append(' '.join(tmp_text[pattern_found[0]: len(attributions[i])]))

            idx_spans.append(tmp_idx_span)
            text_spans.append(tmp_text_span)

        return idx_spans, text_spans

    def calculate_scores(self, attributions, deltas, texts, idx_terms, terms, threshold: float = 0.0):
        scores = []

        idx_interpreted, text_interpreted = self.get_output_from_interpreted(attributions, deltas, texts, threshold)

        print(idx_interpreted)
        print(idx_terms)

        for i in tqdm(range(len(attributions)), desc="Calculate scores"):
            unique_union = []
            interpreted = []
            target = []

            for j in range(len(idx_interpreted[i])):
                idx = idx_interpreted[i][j]
                interpreted.extend(list(range(idx[0], idx[1] + 1)))

            for j in range(len(idx_terms[i])):
                idx = idx_terms[i][j]
                target.extend(list(range(idx[0], idx[1] + 1)))

            unique_union.extend(interpreted)
            unique_union.extend(target)

            max_intersection = list(set(interpreted) & set(target))
            scores.append(len(max_intersection) / len(list(set(unique_union))))

        return np.mean(scores), scores

    def visualize_samples(self, text, max_length: int = 200, label: int = 0, target: int = 0):
        if isinstance(text, list):
            for i in range(len(text)):
                attribution, delta, pred_prob, pred, rtext = self.interpret_sample(text[i], max_length, label[i], target[i])
                self.captum_visualizer.add_attributions_to_visualizer(
                    attribution, rtext, pred_prob, pred, label[i], target[i], delta, self.visual_data_record)
        else:
            attribution, delta, pred_prob, pred, rtext = self.interpret_sample(text, max_length, label, target)
            self.captum_visualizer.add_attributions_to_visualizer(
                attribution, rtext, pred_prob, pred, label, target, delta, self.visual_data_record)

        visualization.visualize_text(self.visual_data_record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='ldl', type=str,
                        help='Type of explanations algorithm')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Device')

    args = parser.parse_args()

    captum_interpreter = CaptumInterpreter(
        method=args.type,
        model_path='models/sa_head.pt',
        pretrained_name_or_path='models/bdi-roberta',
        tokenizer_path='models/bdi-tokenizer',
        device=args.device
    )

    (attributions, deltas), texts, (idx_terms, terms) = captum_interpreter.interpret_from_json(json_path='data/data_merged_0308_fixed_capu_fixed_syserr_2.json')

    score, list_score = captum_interpreter.calculate_scores(
        attributions=attributions,
        deltas=deltas,
        texts=texts,
        terms=terms,
        idx_terms=idx_terms,
        threshold=0.0
    )

    print(score)
    print(list_score)

