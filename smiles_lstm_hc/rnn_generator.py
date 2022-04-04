import logging
import time
from functools import total_ordering
from typing import List, Set

import numpy as np
import torch
import torch.nn as nn

from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize_list

from .rnn_model import SmilesRnn
from .rnn_sampler import SmilesRnnSampler
from .rnn_trainer import SmilesRnnTrainer
from .rnn_utils import get_tensor_dataset, load_smiles_from_list
import copy
import random
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Function to compute whether similarity between two molecules is higher than a specified threshold 
def similarity(smiles, other_smiles, threshold=0.4, radius=2, nBits=1024):
    if Chem.MolFromSmiles(smiles) and Chem.MolFromSmiles(other_smiles):
        fp = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=radius, nBits=nBits)
        other_fp = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(other_smiles), radius=radius, nBits=nBits)
        return DataStructs.FingerprintSimilarity(fp, other_fp)>threshold
    else:
        return 0.0


@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score

    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)

    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)


class SmilesRnnMoleculeGenerator:
    """
    character-based RNN language model optimized by with hill-climbing
    """

    def __init__(self, model: SmilesRnn, max_len: int, device: str) -> None:
        """
        Args:
            model: Pre-trained SmilesRnn
            max_len: maximum SMILES length
            device: 'cpu' | 'cuda'
        """

        self.device = device
        self.model = model

        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.sampler = SmilesRnnSampler(device=self.device, batch_size=512)
        self.max_len = max_len
        self.trainer = SmilesRnnTrainer(model=self.model,
                                        criteria=[self.criterion],
                                        optimizer=self.optimizer,
                                        device=self.device)

    def optimise(self, objective: ScoringFunction, start_population, keep_top, n_epochs, mols_to_sample,
                 optimize_n_epochs, optimize_batch_size, pretrain_n_epochs, beta, threshold, n_diversity=50) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param objective: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param kwargs need to contain:
                keep_top: number of molecules to keep at each iterative finetune step
                mols_to_sample: number of molecules to sample at each iterative finetune step
                optimize_n_epochs: number of episodes to finetune
                optimize_batch_size: batch size for fine-tuning
                pretrain_n_epochs: number of epochs to pretrain on start population
        :return: Candidate molecules
        """
        
        
        int_results = self.pretrain_on_initial_population(objective, list(start_population),
                                                          pretrain_epochs=pretrain_n_epochs)

        results: List[OptResult] = []
        seen: Set[str] = set()
        smiles_history = []
        for k in int_results:
            if k.smiles not in seen:
                results.append(k)
                seen.add(k.smiles)

        for epoch in range(1, 1 + n_epochs):

            t0 = time.time()
            samples = self.sampler.sample(self.model, mols_to_sample, max_seq_len=self.max_len)
            t1 = time.time()

            canonicalized_samples = set(canonicalize_list(samples, include_stereocenters=False))
            payload = list(canonicalized_samples.difference(seen))
            payload.sort()  # necessary for reproducibility between different runs

            seen.update(canonicalized_samples)

            scores = objective.score_list(payload)

            int_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(payload, scores)]
            results.extend(sorted(int_results, reverse=True)[0:keep_top])
            results.sort(reverse=True)
            candidates = [i.smiles for i in results][:keep_top]
            
            # Bring in previous molecules to increase stability
            candidates.extend([i.smiles for i in random.sample(results, keep_top)])
            candidates_scores = scores = objective.score_list(candidates)

            previous_scores = candidates_scores
            
            for i, s in enumerate(previous_scores):
                smiles = candidates[i]
                average_sim = 0
                average_score = 0 
                for j in random.sample(range(len(candidates)), n_diversity):
                    other_smiles = candidates[j]
                    average_sim += previous_scores[j] * similarity(smiles, other_smiles, threshold)
                candidates_scores[i] = s * max(0, (1 - beta * average_sim/n_diversity))

            train_results = [OptResult(smiles=smiles, score=score) for smiles, score in zip(candidates, candidates_scores)]
            train_results.sort(reverse=True)
            t2 = time.time()

           
              
            smiles_history.append([i.smiles for i in train_results][0:keep_top])
            

            
            subset = [i.smiles for i in train_results][:keep_top]
            subset_scores = [i.score for i in train_results][:keep_top]
            
            
            if len(subset)>10:
                
                while len(subset)<10:
                    subset = np.concatenate((subset, subset, subset))
                np.random.shuffle(subset)

                sub_train = subset[0:int(3 * len(subset) / 4)]
                sub_test = subset[int(3 * len(subset) / 4):]

                train_seqs, _ = load_smiles_from_list(sub_train, max_len=self.max_len)
                valid_seqs, _ = load_smiles_from_list(sub_test, max_len=self.max_len)

                train_set = get_tensor_dataset(train_seqs)
                valid_set = get_tensor_dataset(valid_seqs)

                opt_batch_size = min(len(sub_train), optimize_batch_size)

                print_every = int(len(sub_train) / opt_batch_size)
                try:
                    if optimize_n_epochs > 0:
                        self.trainer.fit(train_set, valid_set,
                                     n_epochs=optimize_n_epochs,
                                     batch_size=opt_batch_size,
                                     print_every=print_every,
                                     valid_every=print_every)
                except:
                    print(subset)
            t3 = time.time()

            print(f'Generation {epoch} --- timings: '
                        f'sample: {(t1 - t0):.3f} s, '
                        f'score: {(t2 - t1):.3f} s, '
                        f'finetune: {(t3 - t2):.3f} s')

            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in results[:10])

            print(f'Best optimized:\n{top4}')
            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in train_results[:10])
            print(f'Best selected:\n{top4}')
            top4 = '\n'.join(f'\t{result.score:.3f}: {result.smiles}' for result in results[keep_top-10:keep_top])            

        return sorted(results, reverse=True), smiles_history

    def sample(self, num_mols) -> List[str]:
        """

        :return: a list of molecules
        """

        return self.sampler.sample(self.model,
                                   num_to_sample=num_mols,
                                   max_seq_len=self.max_len)

    # TODO refactor, still has lots of duplication
    def pretrain_on_initial_population(self, scoring_function: ScoringFunction,
                                       start_population, pretrain_epochs) -> List[OptResult]:
        """
        Takes an objective and tries to optimise it
        :param scoring_function: MPO
        :param start_population: Initial compounds (list of smiles) or request new (random?) population
        :param pretrain_epochs: number of epochs to finetune with start_population
        :return: Candidate molecules
        """
        seed: List[OptResult] = []

        start_population_size = len(start_population)

        training = canonicalize_list(start_population, include_stereocenters=True)
        print(training)
        if len(training) != start_population_size:
            logger.warning("Some entries for the start population are invalid or duplicated")
            start_population_size = len(training)

        if start_population_size == 0:
            return seed

        logger.info("finetuning with {} molecules for {} epochs".format(start_population_size, pretrain_epochs))
        print("finetuning with {} molecules for {} epochs".format(start_population_size, pretrain_epochs))
        scores = scoring_function.score_list(training)
        seed.extend(OptResult(smiles=smiles, score=score) for smiles, score in zip(training, scores))

        train_seqs, _ = load_smiles_from_list(training, max_len=self.max_len)
        train_set = get_tensor_dataset(train_seqs)

        batch_size = min(int(len(training)), 32)

        print_every = len(training) / batch_size

        losses = self.trainer.fit(train_set, train_set,
                                  batch_size=batch_size,
                                  n_epochs=pretrain_epochs,
                                  print_every=print_every,
                                  valid_every=print_every)
        logger.info(losses)
        return seed

