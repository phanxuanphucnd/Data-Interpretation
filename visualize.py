# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan


import torch
import numpy as np
from typing import Iterable
from constants import ID2LABEL, LABEL2ID
try:
    from IPython.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_class",
        "attr_score",
        "raw_input_ids",
        "convergence_score",
    ]

    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input_ids,
        convergence_score,
    ) -> None:
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input_ids = raw_input_ids
        self.convergence_score = convergence_score


class CaptumVisualizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_attributions_to_visualizer(self, attributions, text, pred_prob, pred, label, target, delta, vis_data_records):
        # attributions = attributions.sum(dim=2).squeeze(0)
        # attributions = attributions / torch.norm(attributions)
        # attributions = attributions.cpu().detach().numpy()

        text = [self.tokenizer.decode(i) for i in text[0]]

        label_dict = ID2LABEL
        # storing couple samples in an array for visualization purposes
        ''' "word_attributions",
            "pred_prob",
            "pred_class",
            "true_class",
            "attr_class",
            "attr_score",
            "raw_input_ids",
            "convergence_scor" '''
            
        keep_id = []
        for i in range(len(text)):
            if text[i] not in ['<s>', '<pad>', '</s>']:
                keep_id.append(i)

        attributions = np.array([attributions[i] for i in keep_id])

        text = [text[i] for i in keep_id]

        print(f"attributions: {attributions} \n text: {text}")
        vis_data_records.append(
            VisualizationDataRecord(
                attributions,
                pred_prob,
                label_dict[pred],
                label_dict[label],
                label_dict[target],
                attributions.sum(),
                text,
                delta
            )
        )

    def visualize_text(
        self, datarecords: Iterable[VisualizationDataRecord], legend: bool = True
    ) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
        assert HAS_IPYTHON, (
            "IPython must be available to visualize text. "
            "Please run 'pip install ipython'."
        )
        dom = ["<table width: 100%>"]
        rows = [
            "<tr><th>True Label</th>"
            "<th>Predicted Label</th>"
            "<th>Attribution Label</th>"
            "<th>Attribution Score</th>"
            "<th>Word Importance</th>"
        ]
        for datarecord in datarecords:
            rows.append(
                "".join(
                    [
                        "<tr>",
                        self.format_classname(datarecord.true_class),
                        self.format_classname(
                            "{0} ({1:.2f})".format(
                                datarecord.pred_class, datarecord.pred_prob
                            )
                        ),
                        self.format_classname(datarecord.attr_class),
                        self.format_classname("{0:.2f}".format(datarecord.attr_score)),
                        self.format_word_importances(
                            datarecord.raw_input_ids, datarecord.word_attributions
                        ),
                        "<tr>",
                    ]
                )
            )

        if legend:
            dom.append(
                '<div style="border-top: 1px solid; margin-top: 5px; \
                padding-top: 5px; display: inline-block">'
            )
            dom.append("<b>Legend: </b>")

            for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
                dom.append(
                    '<span style="display: inline-block; width: 10px; height: 10px; \
                    border: 1px solid; background-color: \
                    {value}"></span> {label}  '.format(
                        value=self._get_color(value), label=label
                    )
                )
            dom.append("</div>")

        dom.append("".join(rows))
        dom.append("</table>")
        html = HTML("".join(dom))
        display(html)

        return html
    
    def format_classname(self, classname):
        return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)

    def _get_color(self, attr):
        # clip values to prevent CSS errors (Values should be from [-1,1])
        attr = max(-1, min(1, attr))
        if attr > 0:
            hue = 120
            sat = 75
            lig = 100 - int(50 * attr)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * attr)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)

    def format_word_importances(self, words, importances):
        if importances is None or len(importances) == 0:
            return "<td></td>"
        assert len(words) <= len(importances)
        tags = ["<td>"]
        for word, importance in zip(words, importances[: len(words)]):
            word = self.format_special_tokens(word)
            color = self._get_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                color=color, word=word
            )
            tags.append(unwrapped_tag)
        tags.append("</td>")
        return "".join(tags)

    def format_special_tokens(self, token):
        if token.startswith("<") and token.endswith(">"):
            return "#" + token.strip("<>")
        return token

